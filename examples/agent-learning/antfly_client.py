# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The handful of Antfly HTTP calls this example needs.

Only the standard library is used so the example runs anywhere Python does.
Swap this for the Antfly Python SDK in a real application.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8080"


class AntflyError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(f"antfly returned HTTP {status}: {body}")
        self.status = status
        self.body = body


class Antfly:
    def __init__(self, url=None, token=None, timeout=30):
        self.url = (url or os.environ.get("ANTFLY_URL") or DEFAULT_URL).rstrip("/")
        self.token = token or os.environ.get("ANTFLY_TOKEN")
        self.timeout = timeout

    def _request(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as err:
            raise AntflyError(err.code, err.read().decode(errors="replace")) from None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return raw.decode(errors="replace")

    def create_table(self, name, keyword_fields=(), keyword_lists=()):
        """Create a schemaless table, declaring identifier fields as keywords.

        Text fields are analyzed, so `refunds_v2` would also match `refunds`.
        Keyword fields match whole values only, which is what filters need.
        """
        props = {
            f: {"type": "string", "x-antfly-types": ["keyword"]} for f in keyword_fields
        }
        for f in keyword_lists:
            props[f] = {
                "type": "array",
                "items": {"type": "string"},
                "x-antfly-types": ["keyword"],
            }
        schema = {
            "type": "object",
            "additionalProperties": True,
            "x-antfly-dynamic-indexing": {"mode": "infer_types"},
            "properties": props,
        }
        body = {
            "schema": {
                "default_type": "doc",
                "document_schemas": {"doc": {"schema": schema}},
            }
        }
        return self._request("POST", f"/db/v1/tables/{_quote(name)}", body)

    def drop_table(self, name):
        try:
            self._request("DELETE", f"/db/v1/tables/{_quote(name)}")
        except AntflyError as err:
            if err.status != 404:
                raise

    def put(self, table, docs):
        """Insert or overwrite documents. `docs` maps key -> document."""
        if not docs:
            return
        self._request(
            "POST",
            f"/db/v1/tables/{_quote(table)}/batch",
            {"inserts": docs, "sync_level": "full_text"},
        )

    def get(self, table, key):
        """Fetch one document by key, or None if it doesn't exist."""
        try:
            doc = self._request(
                "GET", f"/db/v1/tables/{_quote(table)}/documents/{_quote(key)}"
            )
        except AntflyError as err:
            if err.status == 404:
                return None
            raise
        return dict(doc, _id=key)

    def search(self, table, query, limit=10, offset=0, fields=None):
        """Run a full-text query and return the hits as plain documents.

        Each returned document carries its key as `_id` and its relevance
        as `_score`.
        """
        body = {"table": table, "limit": limit, "offset": offset}
        if query:
            body["full_text_search"] = {"query": query}
        if fields:
            body["fields"] = fields
        resp = self._request("POST", "/db/v1/query", body)
        result = resp["responses"][0]
        if result.get("status", 200) != 200:
            raise AntflyError(result.get("status"), result.get("error"))
        hits = []
        for hit in result["hits"]["hits"]:
            doc = dict(hit.get("_source") or {})
            doc["_id"] = hit["_id"]
            doc["_score"] = hit.get("_score", 0.0)
            hits.append(doc)
        return hits

    def scan(self, table, query=None, page_size=200):
        """Return every document that matches `query`, paging through results."""
        docs, offset = [], 0
        while True:
            page = self.search(table, query, limit=page_size, offset=offset)
            docs.extend(page)
            if len(page) < page_size:
                return docs
            offset += page_size


def _quote(name):
    return urllib.parse.quote(name, safe="")
