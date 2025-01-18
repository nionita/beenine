// C implementation of vector multiply with AVX2:
#include <immintrin.h>
#include <stdint.h>

void multiply_vectors_avx2(const int8_t* a, const int8_t* b, int16_t* result, size_t length) {
    for (size_t i = 0; i < length; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&b[i]);
        
        __m256i low = _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 0)),
                                         _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 0)));
        __m256i high = _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1)),
                                          _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1)));
        
        _mm256_storeu_si256((__m256i*)&result[i], low);
        _mm256_storeu_si256((__m256i*)&result[i + 16], high);
    }
}

// Haskell FFI
{-# LANGUAGE ForeignFunctionInterface #-}

import Foreign.Ptr
import Foreign.C.Types
import qualified Data.Vector.Storable as V
import Foreign.Marshal.Array

foreign import ccall unsafe "multiply_vectors_avx2"
    c_multiply_vectors_avx2 :: Ptr CChar -> Ptr CChar -> Ptr CShort -> CSize -> IO ()

multiplyVectorsAVX2 :: V.Vector Int8 -> V.Vector Int8 -> V.Vector Int16
multiplyVectorsAVX2 v1 v2
    | V.length v1 /= V.length v2 = error "Vectors must have the same length"
    | V.length v1 `mod` 32 /= 0 = error "Vector length must be a multiple of 32"
    | otherwise = V.create $ do
        result <- V.new (V.length v1)
        V.unsafeWith v1 $ \p1 ->
            V.unsafeWith v2 $ \p2 ->
                V.unsafeWith result $ \pr ->
                    c_multiply_vectors_avx2 p1 p2 pr (fromIntegral $ V.length v1)
        return result

// Haskell usage:
import qualified Data.Vector.Storable as V

main :: IO ()
main = do
    let v1 = V.fromList [1, 2, 3, 4, 5, 6, 7, 8] :: V.Vector Int8
    let v2 = V.fromList [8, 7, 6, 5, 4, 3, 2, 1] :: V.Vector Int8
    let result = multiplyVectorsAVX2 v1 v2
    print result