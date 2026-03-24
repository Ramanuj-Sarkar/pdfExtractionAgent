# pdfExtractionAgent
This agent extracts information from PDFs.

# How to Run:

Download things in requirements.txt

Go into:

```
../venv/lib/python3.13/site-packages/paddlex/inference/pipelines/components/retriever/base.py
```

and update base.py by replacing:

```
 if is_dep_available("langchain"):
      from langchain.docstore.document import Document
      from langchain.text_splitter import RecursiveCharacterTextSplitter
```

with

```
 if is_dep_available("langchain"):
     from langchain_core.documents import Document
     from langchain_text_splitters import RecursiveCharacterTextSplitter
```

Please bring up a problem if you have a problem running this.