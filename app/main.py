from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

# === Routers ===
from app.api.score_api import router as score_router
from app.api.train_api import router as train_router
from app.api.monitor_api import router as monitor_router
from app.api.review_queue_api import router as review_router

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LenderAI")

# === FastAPI App ===
app = FastAPI(
    title="LenderAI Credit Scoring System",
    description="AI-powered Credit Scoring, Fraud Detection & Review Workflow",
    version="1.0.0"
)

# === CORS (Adjust for production) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Include Routers ===
app.include_router(score_router, prefix="/api/score", tags=["Scoring"])
app.include_router(train_router, prefix="/api/train", tags=["Training"])
app.include_router(monitor_router, prefix="/api/monitor", tags=["Monitoring"])
app.include_router(review_router, prefix="/api/review", tags=["Review Queue"])

# === Health Check ===
@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "LenderAI API is running 🚀"}

@app.get("/ping", tags=["Health"])
async def ping():
    return {"pong": True}

# === Exception Handlers ===
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"[HTTP Error] {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"[Validation Error] {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"error": "Validation Error", "details": exc.errors()}
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"[Unhandled Exception] {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error"}
    )

# === Startup & Shutdown Events ===
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 LenderAI API is starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 LenderAI API is shutting down...")
