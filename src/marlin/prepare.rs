use super::MarlinNvFp4RepackSpec;
use crate::{Context, DeviceBuffer, Error, Result, Stream};

impl Context {
    /// Enqueues a one-time canonical-to-Marlin expert-bank weight conversion.
    pub fn marlin_repack_nvfp4(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4RepackSpec,
        input: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        Ok(self.native.marlin_repack_nvfp4(
            &stream.native,
            spec.native(),
            &input.native,
            &output.native,
        )?)
    }

    /// Enqueues a gate/up concatenation and Marlin repack without an intermediate bank.
    pub fn marlin_repack_nvfp4_pair(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4RepackSpec,
        left: &DeviceBuffer<u8>,
        right: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        Ok(self.native.marlin_repack_nvfp4_pair(
            &stream.native,
            spec.native(),
            &left.native,
            &right.native,
            &output.native,
        )?)
    }

    /// Converts canonical E4M3 group scales and global scales to Marlin's layout.
    #[allow(clippy::too_many_arguments)]
    pub fn marlin_prepare_nvfp4_scales(
        &self,
        stream: &Stream,
        spec: MarlinNvFp4RepackSpec,
        left: &DeviceBuffer<u8>,
        right: Option<&DeviceBuffer<u8>>,
        globals: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<u8>,
        output_globals: &mut DeviceBuffer<f32>,
        maximum: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        if maximum.len() != 1 {
            return Err(Error::InvalidMatmulShape);
        }
        Ok(self.native.marlin_prepare_nvfp4_scales(
            &stream.native,
            spec.native(),
            &left.native,
            right.map(|value| value.native.as_ref()),
            &globals.native,
            &output.native,
            &output_globals.native,
            &maximum.native,
        )?)
    }
}
