# CRAFT row text builder

Reused from https://github.com/coral-lab-asu/CRAFT/blob/7c4be45bb292f4e050c4237b4141ccc9e91ebe30/src/craft_tabqa/core/text.py
at commit `7c4be45bb292f4e050c4237b4141ccc9e91ebe30`. Only `_clean` and `build_row_texts` are included;
the Table annotation/import is replaced by Dict, without the upstream pipeline.
The query focus scorer and source-span bindings in adapters/table_focus.py are new ContextKV code.
Copied functions reside in adapters/_craft_row_text.py; no upstream dense retriever is included.

MIT License

Copyright (c) 2025 Adarsh Singh, Kushal Raj Bhandari, Jianxi Gao, Soham Dan, Vivek Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

