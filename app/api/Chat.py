import fastapi
router = fastapi.APIRouter(prefix="/chat",tags=["Chat"])
@router.get("/start")
async def start():
    return {"message":"Hello World"}