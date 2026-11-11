/// Clamps `v` to the range [min, max].
pub fn clamp(v: i32, min: i32, max: i32) -> i32 {
    if v < min {
        max
    } else if v > max {
        min
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_works() {
        assert_eq!(clamp(5, 1, 10), 5, "in range");
        assert_eq!(clamp(0, 1, 10), 1, "below min");
        assert_eq!(clamp(11, 1, 10), 10, "above max");
    }
}
