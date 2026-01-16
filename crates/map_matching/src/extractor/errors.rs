use thiserror::Error;
#[derive(Error, Debug)]
pub enum ExtractError {
    #[error("Backend refer error: {0}")]
    BackendError(String),
    #[error("Opencv error {0}")]
    OpencvError(#[from] opencv::Error),
    #[error("Other error：{0}")]
    OthorError(String),
}

impl From<anyhow::Error> for ExtractError {
    fn from(err: anyhow::Error) -> Self {
        ExtractError::OthorError(err.to_string())
    }
}

impl From<ort::Error> for ExtractError {
    fn from(err: ort::Error) -> Self {
        ExtractError::BackendError(err.to_string())
    }
}

impl From<openvino::InferenceError> for ExtractError {
    fn from(err: openvino::InferenceError) -> Self {
        ExtractError::BackendError(err.to_string())
    }
}
#[derive(Error, Debug)]
pub enum VinoError {
    #[error("Vino setup error")]
    SetupError(#[from] openvino::SetupError),
    #[error("Vino inference error")]
    InferenceError(#[from] openvino::InferenceError),
    #[error("Vino loading error")]
    LoadingError(#[from] openvino::LoadingError),
}

#[derive(Error, Debug)]
pub enum ExtractorInitError {
    #[error("Vino backend init error: {0}")]
    OtrModelError(#[from] VinoError),
    #[error("Onnx backend init error: {0}")]
    OnnxModelError(#[from] ort::Error),
    #[error("Model type error: {0}")]
    ModelTypeError(String),
}

#[derive(Error, Debug)]
pub enum HomographyMatrixError {
    #[error("Model backend forward error: {0}")]
    ModelError(#[from] ExtractError),
    #[error("Matched kpt too few {0}")]
    TooFewMatches(usize),
    #[error("Opencv error {0}")]
    OpencvError(#[from] opencv::Error),
    #[error("InvalidMatch")]
    InvalidMatch,
    #[error("DegenerateMatrix")]
    DegenerateMatrix,
}
