// Check AVX2 support:
#include <cpuid.h>
#include <stdint.h>

int check_avx2_support() {
    uint32_t eax, ebx, ecx, edx;
    __get_cpuid(7, &eax, &ebx, &ecx, &edx);
    return (ebx & bit_AVX2) != 0;
}

// Haskell definition for fallback:
{-# LANGUAGE ForeignFunctionInterface #-}
module VectorMult 
    ( multiplyVectors
    ) where

import Foreign.C.Types
import System.IO.Unsafe (unsafePerformIO)
import qualified Data.Vector.Storable as V

foreign import ccall unsafe "check_avx2_support"
    c_check_avx2_support :: IO CInt

foreign import ccall unsafe "multiply_vectors_avx2"
    c_multiply_vectors_avx2 :: Ptr CChar -> Ptr CChar -> Ptr CShort -> CSize -> IO ()

data Implementation = AVX2 | Default

avx2Supported :: Bool
avx2Supported = unsafePerformIO $ do
    result <- c_check_avx2_support
    return (result /= 0)

chosenImplementation :: Implementation
chosenImplementation = if avx2Supported then AVX2 else Default

multiplyVectors :: V.Vector Int8 -> V.Vector Int8 -> V.Vector Int16
multiplyVectors v1 v2 = case chosenImplementation of
    AVX2    -> multiplyVectorsAVX2 v1 v2
    Default -> multiplyVectorsDefault v1 v2

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

multiplyVectorsDefault :: V.Vector Int8 -> V.Vector Int8 -> V.Vector Int16
multiplyVectorsDefault v1 v2 = V.zipWith (\a b -> fromIntegral a * fromIntegral b) v1 v2
