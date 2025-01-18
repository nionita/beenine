Procedure to generate training / test data for BeeNiNe

1. Select the starting more or less balanced positions from which to play games

I had a file named open-moves.fen with 25783 position fens (one per line) which I used since years to train Barbarossa
through other methods (with no success so far) - but I think those ones are balanced positions which result by making
4 full moves from the initial positions. I copied this to the input directory (/e/extract/2025/test-1/input).

I also had an older file named 4818922_positions_gm2600.txt with 4818921 fens (adnotated with the end result), I guess
containing positions which appeared in GM games. I ignored the result (per sed) and chunked it with split in parts of
20000 fens each in the input directory.

2. Play games with Stockfish to create adnotated position

I used Stockfish 17 to play games with fixed depth 9 search, to generate adnotated positions as created by the sample
option of c-chess-cli (https://github.com/lucasart/c-chess-cli), rezulting in files with a lot more fens with the
following structure (csv): fen,score,rezult

Because some fens from the input files had wrong 5th field (over 100), which c-chess-cli refused, I had to filter them
by awk.

Then I used commands like this to play games, which takes a lot of time (one command for every input file):

./c-chess-cli.exe -each option.Hash=8 option.Threads=1 depth=9 cmd=/c/Engines/stockfish-17/stockfish-windows-x86-64-avx2.exe \
   -games 19991 -concurrency 2 -openings file=/e/extract/2025/test-1/input/xac.fen \
   -resign count=3 score=700 -draw number=40 count=8 score=10 \
   -sample resolve=y file=/e/extract/2025/test-1/xac.csv format=csv \
   -engine name=sf1 -engine name=sf2

3. Now it would be a good time to filter the csv files for unique positions, although I am not sure what is best - I did
not do this for now. This is not quite trivial, as I even saw positions which differed only by the 50 moves counter, and
the score & rezult where different. Not sure how to proceed with these (if we don't send the 50 moves counter to the network,
this will rezult in unclear effects, but maybe I overestimate this).

4. Transform the csv files to feature and target files with a Haskell program (using same eval structures as BeeNiNe). The
rezults are also csv files, but having the feature numbers for the feature files respective the score & rezult for the
target files.

This step is highly dependent on the network structure (including number of features) of BeeNiNe, which means, we must repeat
it when the features or the network struction change.

5. Transform feature and target files to binary numpy files to be memmapped - in order to have a fast dataset.