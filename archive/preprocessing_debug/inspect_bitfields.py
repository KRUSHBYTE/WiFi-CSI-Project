
word = 0x0A01FFB1

# BCM4366c0 packed CSI fields
exponent_raw = word & 0x3F

imag_magnitude = (word >> 6) & 0x7FF
imag_sign = (word >> 17) & 1

real_magnitude = (word >> 18) & 0x7FF
real_sign = (word >> 29) & 1

# Six-bit signed exponent
exponent = (
    exponent_raw - 64
    if exponent_raw >= 32
    else exponent_raw
)

print("Packed word:", hex(word))

print("Real sign:", real_sign)
print("Real magnitude:", real_magnitude)

print("Imaginary sign:", imag_sign)
print("Imaginary magnitude:", imag_magnitude)

print("Raw exponent:", exponent_raw)
print("Signed exponent:", exponent)
