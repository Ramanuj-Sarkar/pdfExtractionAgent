# pdfExtractionAgent
This agent extracts information from PDFs, including complicated table and chart information.

# How to Run:

Download things in requirements.txt.

Layoutreader was obtained using:

```
git clone https://github.com/ppaanngggg/layoutreader.git
```

This project also had to deal with the following issue: https://github.com/PaddlePaddle/PaddleOCR/issues/16711


I specifically fixed it by going into:

```
../venv/lib/python3.13/site-packages/paddlex/inference/pipelines/components/retriever/base.py
```

and updating base.py by replacing:

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

If you have a problem running this, please bring it up.