use std::io::{self, Write};

use mircuda::Driver;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let mut output = io::stdout().lock();
    for device in driver.devices()? {
        let context = driver.create_context(device)?;
        let info = context.device_info()?;
        writeln!(
            output,
            "device={} name={} compute={}.{} sms={} memory={} pools={} integrated={}",
            info.ordinal,
            info.name,
            info.compute_capability.0,
            info.compute_capability.1,
            info.multiprocessor_count,
            info.total_memory,
            info.memory_pools,
            info.integrated
        )?;
    }
    Ok(())
}
