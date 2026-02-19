//! Unit conversion traits for same-dimension unit compatibility.
//!
//! This module provides traits that enable virolai to work with different units
//! of the same physical dimension, allowing tasks to specify durations in one unit
//! (e.g., seconds) while constraints and scheduling operate on another unit
//! (e.g., days/MJD).

use qtty::{Quantity, Unit};

/// Marker trait for units that share the same physical dimension.
///
/// This trait is automatically implemented for any pair of units where
/// `From::Dim == To::Dim`, enabling compile-time checked conversions.
///
/// # Example
///
/// ```ignore
/// use qtty::{Second, Day};
/// use virolai::units::SameDim;
///
/// // This compiles because Second and Day share the Time dimension
/// fn accepts_same_dim<From, To>()
/// where
///     From: SameDim<To>,
/// {}
///
/// accepts_same_dim::<Second, Day>(); // OK
/// // accepts_same_dim::<Second, Meter>(); // Error: different dimensions
/// ```
pub trait SameDim<To: Unit>: Unit<Dim = To::Dim> {}

// Blanket implementation: any two units with the same dimension satisfy SameDim
impl<From, To> SameDim<To> for From
where
    From: Unit,
    To: Unit<Dim = From::Dim>,
{
}

/// Converts a quantity from one unit to another unit of the same dimension.
///
/// This is a convenience wrapper around `Quantity::to::<T>()` that works
/// with the `SameDim` trait bounds.
///
/// # Example
///
/// ```ignore
/// use qtty::{Quantity, Second, Day};
/// use virolai::units::convert;
///
/// let duration_sec = Quantity::<Second>::new(86400.0);
/// let duration_day: Quantity<Day> = convert(duration_sec);
/// assert!((duration_day.value() - 1.0).abs() < 1e-12);
/// ```
#[inline]
pub const fn convert<From, To>(q: Quantity<From>) -> Quantity<To>
where
    From: SameDim<To>,
    To: Unit,
{
    q.to_const::<To>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use qtty::{Day, Hour, Minute, Second};

    #[test]
    fn test_same_dim_seconds_to_days() {
        let seconds = Quantity::<Second>::new(86400.0);
        let days: Quantity<Day> = convert(seconds);
        assert!((days.value() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_same_dim_days_to_seconds() {
        let days = Quantity::<Day>::new(1.0);
        let seconds: Quantity<Second> = convert(days);
        assert!((seconds.value() - 86400.0).abs() < 1e-9);
    }

    #[test]
    fn test_same_dim_hours_to_minutes() {
        let hours = Quantity::<Hour>::new(2.0);
        let minutes: Quantity<Minute> = convert(hours);
        assert!((minutes.value() - 120.0).abs() < 1e-12);
    }

    #[test]
    fn test_convert_preserves_value_semantics() {
        // 1 day = 24 hours = 1440 minutes = 86400 seconds
        let day = Quantity::<Day>::new(1.0);
        let hour: Quantity<Hour> = convert(day);
        let minute: Quantity<Minute> = convert(hour);
        let second: Quantity<Second> = convert(minute);

        assert!((hour.value() - 24.0).abs() < 1e-12);
        assert!((minute.value() - 1440.0).abs() < 1e-9);
        assert!((second.value() - 86400.0).abs() < 1e-6);
    }
}
