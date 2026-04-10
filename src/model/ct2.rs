use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use half::f16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ct2DataType {
    Float32,
    Int8,
    Int16,
    Int32,
    Float16,
    BFloat16,
}

impl Ct2DataType {
    fn from_id(id: u8) -> Result<Self> {
        match id {
            0 => Ok(Self::Float32),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Int16),
            3 => Ok(Self::Int32),
            4 => Ok(Self::Float16),
            5 => Ok(Self::BFloat16),
            _ => bail!("unsupported CTranslate2 dtype id {id}"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: Ct2DataType,
    pub offset: u64,
    pub nbytes: usize,
}

impl TensorInfo {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Debug)]
pub struct Ct2ModelBin {
    path: PathBuf,
    pub version: u32,
    pub spec_name: String,
    pub revision: u32,
    tensors: BTreeMap<String, TensorInfo>,
}

#[derive(Debug, Clone)]
pub struct TensorDataF32 {
    pub info: TensorInfo,
    pub values: Vec<f32>,
}

impl Ct2ModelBin {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut reader = BufReader::new(
            File::open(&path).with_context(|| format!("failed to open {}", path.display()))?,
        );

        let version = read_u32(&mut reader)?;
        let spec_name = read_string(&mut reader)?;
        let revision = read_u32(&mut reader)?;
        let tensor_count = read_u32(&mut reader)? as usize;
        let mut tensors = BTreeMap::new();

        for _ in 0..tensor_count {
            let name = read_string(&mut reader)?;
            let rank = read_u8(&mut reader)? as usize;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(read_u32(&mut reader)? as usize);
            }
            let dtype = Ct2DataType::from_id(read_u8(&mut reader)?)?;
            let nbytes = read_u32(&mut reader)? as usize;
            let offset = reader.stream_position()?;
            reader.seek(SeekFrom::Current(nbytes as i64))?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    shape,
                    dtype,
                    offset,
                    nbytes,
                },
            );
        }

        let alias_count = read_u32(&mut reader)? as usize;
        for _ in 0..alias_count {
            let _ = read_string(&mut reader)?;
            let _ = read_string(&mut reader)?;
        }

        Ok(Self {
            path,
            version,
            spec_name,
            revision,
            tensors,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("tensor not found in model.bin: {name}"))
    }

    pub fn read_tensor_f32(&self, name: &str) -> Result<TensorDataF32> {
        let info = self.tensor(name)?.clone();
        let mut reader = BufReader::new(
            File::open(&self.path)
                .with_context(|| format!("failed to reopen {}", self.path.display()))?,
        );
        reader.seek(SeekFrom::Start(info.offset))?;
        let mut bytes = vec![0_u8; info.nbytes];
        reader.read_exact(&mut bytes)?;

        let values: Vec<f32> = match info.dtype {
            Ct2DataType::Float32 => bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
            Ct2DataType::Float16 => bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f16::from_bits(bits).to_f32()
                })
                .collect(),
            _ => bail!(
                "tensor {name} has dtype {}, not a float tensor",
                info.dtype.label()
            ),
        };

        if values.len() != info.element_count() {
            bail!(
                "tensor {} element count mismatch: expected {}, got {}",
                info.name,
                info.element_count(),
                values.len()
            );
        }

        Ok(TensorDataF32 { info, values })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

fn read_u8(reader: &mut BufReader<File>) -> Result<u8> {
    let mut buf = [0_u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16(reader: &mut BufReader<File>) -> Result<u16> {
    let mut buf = [0_u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(reader: &mut BufReader<File>) -> Result<u32> {
    let mut buf = [0_u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_string(reader: &mut BufReader<File>) -> Result<String> {
    let byte_len = read_u16(reader)? as usize;
    if byte_len == 0 {
        bail!("invalid zero-length string in model.bin");
    }
    let mut buf = vec![0_u8; byte_len];
    reader.read_exact(&mut buf)?;
    if buf.last() != Some(&0) {
        bail!("model.bin string is missing trailing NUL");
    }
    String::from_utf8(buf[..buf.len() - 1].to_vec())
        .with_context(|| "model.bin contained invalid UTF-8")
}
