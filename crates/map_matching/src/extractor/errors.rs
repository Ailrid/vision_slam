use thiserror::Error;
#[derive(Error, Debug)]
pub enum ForwardError {
    #[error("Backend refer error: {0}")]
    BackendError(String),
    #[error("Opencv error {0}")]
    OpencvError(#[from] opencv::Error),
    #[error("Other error：{0}")]
    OthorError(String),
}

impl From<anyhow::Error> for ForwardError {
    fn from(err: anyhow::Error) -> Self {
        ForwardError::OthorError(err.to_string())
    }
}

impl From<ort::Error> for ForwardError {
    fn from(err: ort::Error) -> Self {
        ForwardError::BackendError(err.to_string())
    }
}

impl From<openvino::InferenceError> for ForwardError {
    fn from(err: openvino::InferenceError) -> Self {
        ForwardError::BackendError(err.to_string())
    }
}
#[derive(Error, Debug)]
pub enum HomographyMatrixError {
    #[error("Model backend forward error: {0}")]
    ModelError(#[from] ForwardError),
    #[error("Matched kpt too few {0}")]
    TooFewMatches(usize),
    #[error("Opencv error {0}")]
    OpencvError(#[from] opencv::Error),
    #[error("InvalidMatch")]
    InvalidMatch,
    #[error("DegenerateMatrix")]
    DegenerateMatrix,
}
