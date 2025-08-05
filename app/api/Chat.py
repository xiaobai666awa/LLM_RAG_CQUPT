import fastapi
router = fastapi.APIRouter(prefix="/chat",tags=["Chat"])
@router.post("/start")
async def start(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}