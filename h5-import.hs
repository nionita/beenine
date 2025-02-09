import Data.HDF5 (IOMode(..), File, Dataset)
import qualified Data.HDF5 as H5

main :: IO ()
main = H5.withFile "model.h5" ReadMode $ \file -> do
    weights <- H5.readDataset file "/dense/weight"
    biases  <- H5.readDataset file "/dense/bias"
    -- Use weights/biases in Haskell NN implementation

