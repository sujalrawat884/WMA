import os
import re
import json
import logging
import sys
import io
from logging.handlers import RotatingFileHandler
from pathlib import Path
import pandas as pd
from datetime import date, datetime, time, timezone, timedelta
from typing import List, Optional, TypedDict, Annotated
from contextlib import asynccontextmanager
from operator import itemgetter

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field, BeforeValidator
from typing_extensions import Annotated as DocAnnotated

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from bson import ObjectId

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from twilio.rest import Client

load_dotenv()


def _safe_text_stream(base_stream):
    """Wrap a text stream with UTF-8 encoding if the current encoding is limited."""
    if not base_stream:
        return base_stream
    encoding = (getattr(base_stream, "encoding", "") or "").lower()
    if encoding != "utf-8" and hasattr(base_stream, "buffer"):
        try:
            return io.TextIOWrapper(
                base_stream.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
        except Exception:
            return base_stream
    return base_stream


def _configure_logging():
    level_name = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    stream_handler = logging.StreamHandler(_safe_text_stream(sys.stdout))
    stream_handler.setFormatter(formatter)
    handlers = [stream_handler]

    log_file = os.getenv("APP_LOG_FILE", "/var/log/badminton-ai/app.log")
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            max_bytes = int(os.getenv("APP_LOG_MAX_BYTES", "1048576"))
            backup_count = int(os.getenv("APP_LOG_BACKUP_COUNT", "5"))
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except Exception as exc:  # pragma: no cover - best-effort logging setup
            stream_handler.stream.write(f"WARNING: Failed to configure file logging at {log_file}: {exc}\n")

    logging.basicConfig(level=level, handlers=handlers, force=True)


_configure_logging()
logger = logging.getLogger("BadmintonApp")

apiKey = os.getenv("GOOGLE_API_KEY") # Google API Key (Auto-filled by environment)
if not apiKey:
    raise RuntimeError("GOOGLE_API_KEY is required for the Badminton AI Manager.")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
TWILIO_CONTENT_TEMPLATE_SID = os.getenv("TWILIO_CONTENT_TEMPLATE_SID")

E164_RE = re.compile(r"^\+\d{6,15}$")


def _normalize_e164(number: str) -> str:
    if not number:
        raise ValueError("Phone number is required")
    raw = number.strip()
    if raw.startswith("whatsapp:"):
        raw = raw.split(":", 1)[1]
    raw = raw.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    if not raw.startswith("+"):
        raise ValueError("Phone number must include country code prefix '+'.")
    digits = "+" + "".join(ch for ch in raw if ch.isdigit())
    if not E164_RE.match(digits):
        raise ValueError(f"Invalid E164 phone number: {number}")
    return digits


def _format_whatsapp_address(number: str) -> str:
    normalized = _normalize_e164(number)
    return f"whatsapp:{normalized}"


def _coerce_utc_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)
    return None


def _epoch_millis(value: Optional[datetime]) -> Optional[int]:
    if value is None:
        return None
    return int(value.timestamp() * 1000)


def _stringify_object_ids(value):
    """Recursively convert any ObjectId instances to strings for JSON responses."""
    if isinstance(value, list):
        return [_stringify_object_ids(v) for v in value]
    if isinstance(value, dict):
        return {k: _stringify_object_ids(v) for k, v in value.items()}
    if isinstance(value, ObjectId):
        return str(value)
    return value


def _is_opted_out(phone_number: str) -> bool:
    if not phone_number:
        return False
    with MongoClient(MONGODB_URL, tz_aware=True, tzinfo=timezone.utc) as client:
        doc = client[DB_NAME][OPTOUT_COLLECTION_NAME].find_one({"phone_number": phone_number})
        if not doc:
            return False
        return not doc.get("removed_at")


def _record_notification(phone_number: str, message_body: str, status: str, meta: Optional[dict] = None):
    payload = {
        "phone_number": phone_number,
        "message_body": message_body,
        "status": status,
        "created_at": datetime.now(timezone.utc),
    }
    if meta:
        payload.update(meta)
    try:
        with MongoClient(MONGODB_URL, tz_aware=True, tzinfo=timezone.utc) as client:
            client[DB_NAME][NOTIFICATION_LOG_COLLECTION_NAME].insert_one(payload)
    except Exception as exc:
        logger.warning(f"Failed to record notification log for {phone_number}: {exc}")


MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "royal_badminton_club")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "court_bookings")
INBOUND_COLLECTION_NAME = os.getenv("INBOUND_COLLECTION_NAME", "whatsapp_inbound")
OPTOUT_COLLECTION_NAME = os.getenv("OPTOUT_COLLECTION_NAME", "notification_opt_outs")
NOTIFICATION_LOG_COLLECTION_NAME = os.getenv("NOTIFICATION_LOG_COLLECTION_NAME", "notification_logs")

motor_client = AsyncIOMotorClient(MONGODB_URL, tz_aware=True, tzinfo=timezone.utc)
db = motor_client[DB_NAME]
bookings_collection = db[COLLECTION_NAME]
inbound_collection = db[INBOUND_COLLECTION_NAME]
opt_out_collection = db[OPTOUT_COLLECTION_NAME]

scheduler = AsyncIOScheduler()


PyObjectId = DocAnnotated[str, BeforeValidator(str)]

class Booking(BaseModel):
    booking_id: str
    booking_date: date
    booking_initiated_at: datetime
    booking_expired_at: datetime
    booking_initiated_at_time_stamp: Optional[int] = None
    booking_expired_at_time_stamp: Optional[int] = None
    initiated_at_time_stamp: Optional[int] = None
    expired_at_time_stamp: Optional[int] = None
    bookingId: Optional[PyObjectId] = Field(default=None, alias="bookingId")
    user_id: Optional[str] = None
    court_id: str
    slot_id: str
    user_membership_id: Optional[str] = None
    slot_pricing_id: Optional[str] = None
    bulk_booking_id: Optional[str] = None
    cartId: Optional[str] = None
    is_reserved: bool = True
    is_locked: bool = True
    stripe_session_id: Optional[str] = None
    stripe_transaction_status: Optional[str] = None
    payment_status: Optional[str] = None
    note: Optional[str] = ""
    booking_type: int = 1
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    country_code: Optional[str] = ""
    iso_country_code: Optional[str] = ""
    phone_number: Optional[str] = None
    is_deleted: int = 0
    currency: str = "CAD"
    total_paid_amount: float = 0.0
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None


class BookingRecord(Booking):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)



@tool
def get_booking_history(lookback_days: int = 30):
    """Return recent bookings as a CSV string for the LLM to analyze."""
    try:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be a positive integer")

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        with MongoClient(MONGODB_URL, tz_aware=True, tzinfo=timezone.utc) as client:
            sync_db = client[DB_NAME]
            cursor = (
                sync_db[COLLECTION_NAME]
                .find({
                    "$or": [
                        {"booking_date": {"$gte": cutoff}},
                        {"date": {"$gte": cutoff}},
                    ]
                })
                .sort([("booking_date", -1), ("date", -1), ("createdAt", -1)])
                .limit(200)
            )
            data = list(cursor)
            
            if not data:
                return "No bookings found in database."
            
            cleaned = []
            for doc in data:
                booking_dt = _coerce_utc_datetime(doc.get("booking_date") or doc.get("date"))
                booking_str = booking_dt.strftime("%Y-%m-%d") if booking_dt else None

                first = (doc.get("first_name") or "").strip()
                last = (doc.get("last_name") or "").strip()
                fallback_name = " ".join(part for part in [first, last] if part).strip()
                user_name = doc.get("user_name") or fallback_name or "Unknown"

                phone = (doc.get("phone_number") or doc.get("whatsapp_number") or doc.get("phone") or "").strip()
                if not phone:
                    raw_country = (doc.get("country_code") or "").strip()
                    digits = (doc.get("phone_number") or "").strip()
                    if raw_country and digits:
                        prefix = raw_country if raw_country.startswith("+") else f"+{raw_country}"
                        phone = f"{prefix}{digits.lstrip('+')}"
                if phone:
                    try:
                        phone = _normalize_e164(phone)
                    except ValueError:
                        phone = phone.strip()
                phone = phone or "Unknown"

                cleaned.append({
                    "user": user_name,
                    "phone": phone,
                    "date": booking_str,
                    "day": pd.to_datetime(booking_str).day_name() if booking_str else "Unknown"
                })
                
            df = pd.DataFrame(cleaned)
            return df.to_csv(index=False)
    except Exception as e:
        return f"Error fetching data: {str(e)}"

@tool
def send_whatsapp_reminder(phone_number: str, message_body: str):
    """Send (or simulate) a WhatsApp reminder using Twilio credentials."""
    logger.info(f"📢 AGENT ACTION: Sending message to {phone_number}")

    try:
        normalized_phone = _normalize_e164(phone_number)
    except ValueError as exc:
        return f"Failed to format phone number: {exc}"

    if _is_opted_out(normalized_phone):
        _record_notification(normalized_phone, message_body, "skipped_opt_out")
        return f"SKIPPED: {normalized_phone} opted out of WhatsApp reminders."
    
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        result = f"SIMULATION: Message '{message_body}' sent to {normalized_phone}"
        _record_notification(normalized_phone, message_body, "simulated")
        return result

    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        variables = json.loads(message_body)
        to_address = _format_whatsapp_address(normalized_phone)
        message = client.messages.create(
              from_=TWILIO_WHATSAPP_NUMBER,
              to=to_address,
              content_sid=TWILIO_CONTENT_TEMPLATE_SID,
              content_variables=json.dumps(variables),
        )
        _record_notification(normalized_phone, message_body, "sent", {"sid": getattr(message, "sid", None)})
        return f"Message sent successfully. SID: {message.sid}"
    except Exception as e:
        _record_notification(normalized_phone, message_body, "error", {"error": str(e)})
        return f"Failed to send message: {str(e)}"

tools = [get_booking_history, send_whatsapp_reminder]
tool_map = {t.name: t for t in tools}

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def reasoner_node(state: AgentState):
    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=apiKey)
    model_with_tools = model.bind_tools(tools)
    return {"messages": [model_with_tools.invoke(state["messages"])]}

def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    results = []
    tool_calls = getattr(last_message, "tool_calls", [])
    for t in tool_calls:
        logger.info(f"🔧 Invoking Tool: {t['name']}")
        if t['name'] in tool_map:
            try:
                res = tool_map[t['name']].invoke(t['args'])
                results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(res)))
            except Exception as e:
                results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=f"Error: {str(e)}"))
    return {"messages": results}

def router(state: AgentState):
    last_msg = state["messages"][-1]
    return "tools" if getattr(last_msg, "tool_calls", None) else END


workflow = StateGraph(AgentState)
workflow.add_node("reasoner", reasoner_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("reasoner")
workflow.add_conditional_edges("reasoner", router, {"tools": "tools", END: END})
workflow.add_edge("tools", "reasoner")
agent_app = workflow.compile()


async def run_daily_streak_check():
    now_local = datetime.now().astimezone()
    today_str = now_local.strftime("%Y-%m-%d")
    day_name = now_local.strftime("%A")
    
    logger.info(f"⏰ STARTING DAILY STREAK CHECK for {today_str} ({day_name})")

    # absentees = await find_regular_absentees(date.today())
    # if absentees:
    #     notify_absentees(absentees, today_str, day_name)
    # else:
    #     logger.info("No absentees found by deterministic check.")
    
    prompt = f"""
    You are the Badminton Club Manager. Today is {today_str} ({day_name}).
    
    Goal: Identify regular players who missed their session today and remind them.
    
        1. Call `get_booking_history` to see recent bookings.
        2. Analyze the data:
            - Identify players who usually play on {day_name}s (e.g. played last 2-3 {day_name}s).
            - Check if they have a booking for TODAY ({today_str}).
        3. For every regular player who skipped today, produce ONLY the variables needed by the approved WhatsApp template:
            - Variable {{1}} = the player's preferred first name.
            - Variable {{2}} = a short session label (weekday last attended date), e.g. "Wednesday (last seen 2025-12-01)".
            - Build a JSON string exactly like this: `{{ "1": "<name>", "2": "<session label>" }}` (double braces here so this f-string renders literal braces).
            - Call `send_whatsapp_reminder` with the player's phone number and that JSON string so Twilio can render the template copy on your behalf. Do NOT include custom text or links outside the template.
    
    If no one missed a streak, just output "No reminders needed."
    """
    
    try:
        result = await agent_app.ainvoke({"messages": [HumanMessage(content=prompt)]})
        logger.info("✅ Daily check complete. Agent response:")
        logger.info(result["messages"][-1].content)
    except Exception as e:
        logger.error(f"❌ Error running daily check: {e}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Application starting up...")
    
    try:
        await motor_client.admin.command('ping')
        logger.info("✅ Connected to MongoDB")
    except Exception as e:
        logger.error(f"❌ MongoDB Connection Failed: {e}")

    scheduler.add_job(run_daily_streak_check, 'cron', hour=22, minute=0)
    scheduler.start()
    logger.info("⏰ Scheduler started (Job set for 22:00 daily)")
    
    yield
 
    logger.info("🛑 Application shutting down...")
    scheduler.shutdown()
    motor_client.close()

app = FastAPI(title="Badminton AI Manager", lifespan=lifespan)

async def _extract_twilio_payload(request: Request):
    """Return a dict payload + headers from Twilio webhook requests."""
    headers = {k: v for k, v in request.headers.items()}
    try:
        form = await request.form()
        payload = dict(form.multi_items())
    except Exception:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
    if not payload:
        try:
            raw_body = (await request.body()).decode("utf-8", errors="replace")
            payload = {"raw_body": raw_body}
            # Best-effort parse for application/x-www-form-urlencoded bodies when python-multipart is missing.
            from urllib.parse import parse_qs

            parsed = {k: v[0] if isinstance(v, list) and v else v for k, v in parse_qs(raw_body).items()}
            if parsed:
                payload.update(parsed)
        except Exception:
            payload = payload or {}
    return payload, headers

@app.post("/bookings", status_code=status.HTTP_201_CREATED)
async def add_booking(booking: Booking):
    booking_data = booking.dict(by_alias=True)

    booking_date = _coerce_utc_datetime(booking.booking_date)
    booking_data["booking_date"] = booking_date
    booking_data["booking_initiated_at"] = _coerce_utc_datetime(booking.booking_initiated_at)
    booking_data["booking_expired_at"] = _coerce_utc_datetime(booking.booking_expired_at)

    now_utc = datetime.now(timezone.utc)
    booking_data["createdAt"] = _coerce_utc_datetime(booking_data.get("createdAt")) or now_utc
    booking_data["updatedAt"] = _coerce_utc_datetime(booking_data.get("updatedAt")) or now_utc

    for field in ("first_name", "last_name", "email", "country_code", "iso_country_code", "phone_number", "note"):
        value = booking_data.get(field)
        if isinstance(value, str):
            booking_data[field] = value.strip()

    phone_value = booking_data.get("phone_number")
    if phone_value:
        try:
            normalized_phone = _normalize_e164(phone_value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        booking_data["phone_number"] = normalized_phone
        booking_data["whatsapp_number"] = normalized_phone

    timestamp_pairs = [
        ("booking_initiated_at_time_stamp", "booking_initiated_at"),
        ("booking_expired_at_time_stamp", "booking_expired_at"),
        ("initiated_at_time_stamp", "booking_initiated_at"),
        ("expired_at_time_stamp", "booking_expired_at"),
    ]
    for ts_field, source in timestamp_pairs:
        if not booking_data.get(ts_field):
            booking_data[ts_field] = _epoch_millis(booking_data.get(source))

    booking_data.setdefault("currency", "CAD")
    booking_data.setdefault("total_paid_amount", 0.0)
    booking_data.setdefault("is_deleted", 0)

    if booking_data.get("bookingId") is None:
        booking_data.pop("bookingId", None)

    res = await bookings_collection.insert_one(booking_data)
    return {"id": str(res.inserted_id), "message": "Booking confirmed"}

@app.get("/bookings")
async def get_bookings():
    bookings = (
        await bookings_collection
        .find()
        .sort([("booking_date", -1), ("date", -1), ("createdAt", -1)])
        .to_list(100)
    )
    summarized = []
    for doc in bookings:
        name = (
            (doc.get("user_name") or "").strip()
            or " ".join(filter(None, [(doc.get("first_name") or "").strip(), (doc.get("last_name") or "").strip()])).strip()
            or "Unknown"
        )
        booking_dt = _coerce_utc_datetime(doc.get("booking_date") or doc.get("date"))
        summarized.append({
            "name": name,
            "booking_date": booking_dt.isoformat() if booking_dt else None,
            "court_id": _stringify_object_ids(doc.get("court_id")),
            "slot_id": _stringify_object_ids(doc.get("slot_id")),
        })

    return summarized

@app.post("/admin/trigger-check")
async def manual_trigger():

    scheduler.add_job(
        run_daily_streak_check,
        trigger="date",
        run_date=datetime.now(timezone.utc)
    )
    return {"message": "Agent execution triggered in background."}

@app.post("/webhook/whatsapp/inbound", status_code=status.HTTP_200_OK)
async def whatsapp_inbound(request: Request):
    payload, headers = await _extract_twilio_payload(request)
    logger.info("📥 Incoming WhatsApp message: headers=%s payload=%s", headers, payload)
    waid = payload.get("WaId") or payload.get("waid")
    if not waid:
        from_field = payload.get("From") or ""
        if "whatsapp:" in from_field:
            waid = from_field.split(":", 1)[1]
    doc = {
        "payload": payload,
        "headers": headers,
        "from": payload.get("From"),
        "to": payload.get("To"),
        "body": payload.get("Body") or payload.get("raw_body"),
        "message_sid": payload.get("MessageSid"),
        "WaId": waid,
        "received_at": datetime.now(timezone.utc),
    }
    try:
        await inbound_collection.insert_one(doc)
    except Exception as exc:
        logger.error("Failed to persist inbound WhatsApp message: %s", exc)
    return {"status": "received"}

@app.post("/webhook/whatsapp/status", status_code=status.HTTP_200_OK)
async def whatsapp_status_callback(request: Request):
    payload, headers = await _extract_twilio_payload(request)
    logger.info("📊 WhatsApp delivery status: headers=%s payload=%s", headers, payload)
    return {"status": "acknowledged"}


@app.get("/inbound")
async def list_inbound_messages(waid: Optional[str] = None):
    """Return inbound WhatsApp messages filtered by WaId, only WaId and body fields."""
    if not waid:
        raise HTTPException(status_code=400, detail="Query param 'waid' is required")

    docs = (
        await inbound_collection
        .find({"$or": [{"WaId": waid}, {"payload.WaId": waid}]})
        .sort([("received_at", -1)])
        .limit(50)
        .to_list(None)
    )
    simplified = []
    for d in docs:
        simplified.append({
            "waid": d.get("WaId") or d.get("payload", {}).get("WaId"),
            "body": d.get("body") or d.get("payload", {}).get("Body") or d.get("payload", {}).get("raw_body"),
        })
    return simplified

@app.get("/")
async def root():
    jobs = scheduler.get_jobs()
    next_run = "None"
    if jobs:
        next_time = jobs[0].next_run_time
        next_run = next_time.isoformat() if next_time else "None"
    return {
        "status": "online",
        "scheduler": "running" if scheduler.running else "stopped",
        "next_run": next_run
    }

# To run: uvicorn main:app --reload
