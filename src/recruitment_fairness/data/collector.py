# src/recruitment_fairness/data/collector.py

import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


class ClinicalTrialsWebCollector:
    def __init__(self, output_dir="data/raw"):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.rate_limit = 1.2  # seconds between requests
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def search_trials(self, query_term: str, max_studies: int = 500) -> pd.DataFrame:
        """
        Fetch up to `max_studies` matching `query_term` from the v2 JSON API,
        write a timestamped CSV, and return the DataFrame.
        """
        all_studies, page_token, page_size = [], None, 100
        pbar = tqdm(total=max_studies, desc=f"Fetching '{query_term or 'all'}'")
        while len(all_studies) < max_studies:
            params = {
                "format": "json",
                "pageSize": min(page_size, max_studies - len(all_studies)),
            }
            # use the `query.cond` param to filter by condition:
            if query_term:
                params["query.cond"] = query_term
            if page_token:
                params["pageToken"] = page_token

            r = requests.get(self.base_url, params=params, timeout=30)
            try:
                r.raise_for_status()
            except Exception:
                print(f"⚠️ API error [{r.status_code}]: {r.text[:200]}…")
                break

            data = r.json()
            studies = data.get("studies", [])
            if not studies:
                break

            # extract only the fields you care about
            processed = [self._extract_fields(s) for s in studies]
            all_studies.extend([rec for rec in processed if rec])
            pbar.update(len(studies))

            page_token = data.get("nextPageToken")
            if not page_token:
                break

            time.sleep(self.rate_limit)

        pbar.close()
        df = pd.DataFrame(all_studies[:max_studies])
        # write out a timestamped CSV
        if not df.empty:
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            out_file = (
                self.output_dir / f"raw_clinical_trials_{query_term or 'all'}_{ts}.csv"
            )
            df.to_csv(out_file, index=False)
            print(f"✅ {len(df)} rows saved to {out_file}")
        else:
            print("❌ No data collected.")
        return df

    def _extract_fields(self, study: dict) -> dict:
        """Flatten a single 'study' record into your desired columns."""
        try:
            protocol = study.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            #            desc = protocol.get("descriptionModule", {})
            status = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            #            info = design.get("designInfo", {})
            enroll = design.get("enrollmentInfo", {})

            return {
                "nct_id": ident.get("nctId", ""),
                "brief_title": ident.get("briefTitle", ""),
                "overall_status": status.get("overallStatus", ""),
                "start_date": status.get("startDateStruct", {}).get("date", ""),
                "phases": "|".join(design.get("phases", [])),
                "enrollment_count": enroll.get("count", 0),
                # add more fields as you need…
            }
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return None
