from fastapi import FastAPI
from service.api.api import main_router
import onnxruntime as rt
import uvicorn

app = FastAPI(project_name="Emotions Detection")
app.include_router(main_router)

model_path = "service/eff_quantized.onnx"
providers = ['CPUExecutionProvider']
m_q = rt.InferenceSession(
    model_path, providers=providers
)


@app.get("/")
async def root():
    return {'hello': 'world'}
