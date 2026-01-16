use crate::extractor::errors::ExtractorInitError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CropError {
    #[error("服务器错误: {0}")]
    HttpError(#[from] reqwest::Error),
    #[error("转换Mat错误: {0}")]
    ConvertError(#[from] opencv::Error),
}

pub type SearchError = reqwest::Error;

#[derive(Error, Debug)]
pub enum MatcherError {
    #[error("后端计算错误: {0}")]
    BackendError(String),
    #[error("其他错误：{0}")]
    OthorError(String),
}

impl From<anyhow::Error> for MatcherError {
    fn from(err: anyhow::Error) -> Self {
        MatcherError::OthorError(err.to_string())
    }
}
impl From<opencv::Error> for MatcherError {
    fn from(err: opencv::Error) -> Self {
        MatcherError::OthorError(err.to_string())
    }
}

impl From<ort::Error> for MatcherError {
    fn from(err: ort::Error) -> Self {
        MatcherError::BackendError(err.to_string())
    }
}

impl From<openvino::InferenceError> for MatcherError {
    fn from(err: openvino::InferenceError) -> Self {
        MatcherError::BackendError(err.to_string())
    }
}

pub type MatcherInitError = ExtractorInitError;
