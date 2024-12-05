## Build
Build with docker compose
```bash
docker compose build
```

## Run - Step 1
Start the Ollama service (make sure you don't have running elsewhere).
This is pull down the necessary model too
```bash
docker compose up ollama
```
## Run - Step 2
Running the app component manually at the moment for testing, though this could be automated to read from a source directory of pdfs, json csvs etc.
Running the rag app in host network mode, mapping in some sample pdf files via a bind mount, then reading and querying in the app
```bash
docker run run -it --net host  -v /<source of pdfs>:/data rag-container-app bash 
python main.py /data/doc1.pdf /data/doc2.pdf ...
```


