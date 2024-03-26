#!/usr/bin/env bash

DIR=$1

mkdir -p "$DIR"

# Patent Citation Network (CP)
# Number of Vertices:  6.0 M
# Number of Edges   : 16.5 M
wget -P "$DIR" https://snap.stanford.edu/data/cit-Patents.txt.gz
mv "$DIR/cit-Patents.txt.gz" "$DIR/cit-patents.txt.gz"
gunzip "$DIR/cit-patents.txt.gz"
SNAPtoAdj "$DIR/cit-patents.txt" "$DIR/cit-patents"

# LiveJournal Social Network (LJ)
# Number of Vertices:  4.8 M
# Number of Edges   : 69.0 M
wget -P "$DIR" https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
mv "$DIR/soc-LiveJournal1.txt.gz" "$DIR/soc-livejournal1.txt.gz"
gunzip "$DIR/soc-livejournal1.txt.gz"
SNAPtoAdj "$DIR/soc-livejournal1.txt" "$DIR/soc-livejournal1"

# Pokec Social Network (PS)
# Number of Vertices:  1.6 M
# Number of Edges   : 30.6 M
wget -P "$DIR" https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz
gunzip "$DIR/soc-pokec-relationships.txt.gz"
SNAPtoAdj "$DIR/soc-pokec-relationships.txt" "$DIR/soc-pokec-relationships"

# Wikipedia Talk Network (WT)
# Number of Vertices: 2.4 M
# Number of Edges   : 5.0 M
wget -P "$DIR" https://snap.stanford.edu/data/wiki-Talk.txt.gz
mv "$DIR/wiki-Talk.txt.gz" "$DIR/wiki-talk.txt.gz"
gunzip "$DIR/wiki-talk.txt.gz"
SNAPtoAdj "$DIR/wiki-talk.txt" "$DIR/wiki-talk"
