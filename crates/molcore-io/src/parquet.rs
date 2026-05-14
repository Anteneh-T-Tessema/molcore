/// Arrow/Parquet columnar serialization for MolRecord batches.
///
/// Schema (one row per molecule):
///   smiles        : Utf8
///   n_atoms       : Int32
///   n_bonds       : Int32
///   mw            : Float32
///   logp          : Float32
///   heavy_atoms   : Int32
///   fingerprint   : FixedSizeBinary(256)  — 2048-bit ECFP4 packed as 256 bytes
use std::sync::Arc;

use arrow::array::{
    Array, FixedSizeBinaryBuilder, Float32Builder, Int32Builder, StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// A flat record representing one molecule — decoupled from the Rust graph type
/// so this crate does not need to depend on molcore-core.
#[derive(Debug, Clone)]
pub struct MolRecord {
    pub smiles:      String,
    pub n_atoms:     i32,
    pub n_bonds:     i32,
    pub mw:          f32,
    pub logp:        f32,
    pub heavy_atoms: i32,
    /// 256 bytes = 2048 bits of ECFP4 (bit-packed)
    pub fingerprint: Option<[u8; 256]>,
}

impl MolRecord {
    pub fn new(smiles: impl Into<String>) -> Self {
        MolRecord {
            smiles:      smiles.into(),
            n_atoms:     0,
            n_bonds:     0,
            mw:          0.0,
            logp:        0.0,
            heavy_atoms: 0,
            fingerprint: None,
        }
    }
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("smiles",      DataType::Utf8,                   false),
        Field::new("n_atoms",     DataType::Int32,                  false),
        Field::new("n_bonds",     DataType::Int32,                  false),
        Field::new("mw",          DataType::Float32,                false),
        Field::new("logp",        DataType::Float32,                false),
        Field::new("heavy_atoms", DataType::Int32,                  false),
        Field::new("fingerprint", DataType::FixedSizeBinary(256),   true),
    ]))
}

/// Serialize a batch of MolRecord to Parquet bytes (in-memory).
///
/// Uses SNAPPY compression. Call this from Python via PyO3 or use directly in Rust pipelines.
pub fn write_parquet_bytes(records: &[MolRecord]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let sc = schema();

    let mut smiles_b    = StringBuilder::new();
    let mut n_atoms_b   = Int32Builder::new();
    let mut n_bonds_b   = Int32Builder::new();
    let mut mw_b        = Float32Builder::new();
    let mut logp_b      = Float32Builder::new();
    let mut heavy_b     = Int32Builder::new();
    let mut fp_b        = FixedSizeBinaryBuilder::new(256);

    for r in records {
        smiles_b.append_value(&r.smiles);
        n_atoms_b.append_value(r.n_atoms);
        n_bonds_b.append_value(r.n_bonds);
        mw_b.append_value(r.mw);
        logp_b.append_value(r.logp);
        heavy_b.append_value(r.heavy_atoms);
        match &r.fingerprint {
            Some(fp) => fp_b.append_value(fp)?,
            None     => fp_b.append_null(),
        }
    }

    let batch = RecordBatch::try_new(
        sc.clone(),
        vec![
            Arc::new(smiles_b.finish()),
            Arc::new(n_atoms_b.finish()),
            Arc::new(n_bonds_b.finish()),
            Arc::new(mw_b.finish()),
            Arc::new(logp_b.finish()),
            Arc::new(heavy_b.finish()),
            Arc::new(fp_b.finish()),
        ],
    )?;

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buf, sc, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
    }
    Ok(buf)
}

fn _batch_to_records(
    batch: &RecordBatch,
) -> Result<Vec<MolRecord>, Box<dyn std::error::Error>> {
    let n = batch.num_rows();

    macro_rules! col_as {
        ($name:expr, $t:ty) => {
            batch
                .column_by_name($name)
                .and_then(|c| c.as_any().downcast_ref::<$t>())
                .ok_or(concat!("missing column: ", $name))?
        };
    }

    let smiles_col  = col_as!("smiles",      arrow::array::StringArray);
    let n_atoms_col = col_as!("n_atoms",     arrow::array::Int32Array);
    let n_bonds_col = col_as!("n_bonds",     arrow::array::Int32Array);
    let mw_col      = col_as!("mw",          arrow::array::Float32Array);
    let logp_col    = col_as!("logp",        arrow::array::Float32Array);
    let heavy_col   = col_as!("heavy_atoms", arrow::array::Int32Array);
    let fp_col = batch
        .column_by_name("fingerprint")
        .and_then(|c| c.as_any().downcast_ref::<arrow::array::FixedSizeBinaryArray>());

    let mut records = Vec::with_capacity(n);
    for i in 0..n {
        let fingerprint = fp_col.and_then(|col| {
            if col.is_null(i) { return None; }
            let mut arr = [0u8; 256];
            arr.copy_from_slice(col.value(i));
            Some(arr)
        });
        records.push(MolRecord {
            smiles:      smiles_col.value(i).to_string(),
            n_atoms:     n_atoms_col.value(i),
            n_bonds:     n_bonds_col.value(i),
            mw:          mw_col.value(i),
            logp:        logp_col.value(i),
            heavy_atoms: heavy_col.value(i),
            fingerprint,
        });
    }
    Ok(records)
}

/// Deserialize Parquet bytes back into MolRecord structs.
pub fn read_parquet_bytes(bytes: &[u8]) -> Result<Vec<MolRecord>, Box<dyn std::error::Error>> {
    let owned   = Bytes::copy_from_slice(bytes);
    let builder = ParquetRecordBatchReaderBuilder::try_new(owned)?;
    let mut reader = builder.build()?;
    let mut records = Vec::new();
    for batch_result in &mut reader {
        records.extend(_batch_to_records(&batch_result?)?);
    }
    Ok(records)
}

// ---------------------------------------------------------------------------
// File-level convenience wrappers
// ---------------------------------------------------------------------------

pub fn write_parquet_file(
    records: &[MolRecord],
    path: impl AsRef<std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = write_parquet_bytes(records)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn read_parquet_file(
    path: impl AsRef<std::path::Path>,
) -> Result<Vec<MolRecord>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = builder.build()?;
    let mut records = Vec::new();
    for batch_result in &mut reader {
        let batch = batch_result?;
        records.extend(_batch_to_records(&batch)?);
    }
    Ok(records)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_records() -> Vec<MolRecord> {
        vec![
            MolRecord { smiles: "CCO".into(),       n_atoms: 3, n_bonds: 2, mw: 46.07, logp: -0.31, heavy_atoms: 3, fingerprint: None },
            MolRecord { smiles: "c1ccccc1".into(),  n_atoms: 6, n_bonds: 6, mw: 78.11, logp:  1.56, heavy_atoms: 6, fingerprint: Some([1u8; 256]) },
            MolRecord { smiles: "CC(=O)O".into(),   n_atoms: 4, n_bonds: 3, mw: 60.05, logp: -0.17, heavy_atoms: 4, fingerprint: None },
        ]
    }

    #[test]
    fn round_trip_bytes() {
        let records = sample_records();
        let bytes   = write_parquet_bytes(&records).unwrap();
        assert!(!bytes.is_empty(), "parquet bytes must be non-empty");
        let recovered = read_parquet_bytes(&bytes).unwrap();
        assert_eq!(recovered.len(), records.len());
        for (orig, rec) in records.iter().zip(recovered.iter()) {
            assert_eq!(orig.smiles, rec.smiles);
            assert_eq!(orig.n_atoms, rec.n_atoms);
            assert!((orig.mw - rec.mw).abs() < 0.01, "mw round-trip");
        }
    }

    #[test]
    fn fingerprint_round_trip() {
        let mut r    = MolRecord::new("c1ccccc1");
        r.fingerprint = Some([42u8; 256]);
        let bytes    = write_parquet_bytes(&[r.clone()]).unwrap();
        let recs     = read_parquet_bytes(&bytes).unwrap();
        assert_eq!(recs[0].fingerprint, Some([42u8; 256]));
    }

    #[test]
    fn null_fingerprint_round_trip() {
        let r     = MolRecord::new("CCO");
        let bytes = write_parquet_bytes(&[r]).unwrap();
        let recs  = read_parquet_bytes(&bytes).unwrap();
        assert!(recs[0].fingerprint.is_none());
    }

    #[test]
    fn empty_batch() {
        let bytes = write_parquet_bytes(&[]).unwrap();
        let recs  = read_parquet_bytes(&bytes).unwrap();
        assert!(recs.is_empty());
    }
}
