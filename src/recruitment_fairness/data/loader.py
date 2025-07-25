# src/recruitment_fairness/data/loader.py

import json
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


class ClinicalTrialsWebCollector:
    def __init__(self, output_dir="data/raw"):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.rate_limit = 1.2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def search_trials(self, query_term="", max_studies=5000):
        all_studies, page_token, page_size = [], None, 1000
        pbar = tqdm(total=max_studies, desc=f"Fetching {query_term or 'all'}")
        while len(all_studies) < max_studies:
            try:
                params = {
                    "format": "json",
                    "pageSize": min(page_size, max_studies - len(all_studies)),
                }
                if query_term:
                    params["query.cond"] = query_term
                if page_token:
                    params["pageToken"] = page_token
                r = requests.get(self.base_url, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                if "studies" not in data or not data["studies"]:
                    break
                processed = [self._extract_fields(study) for study in data["studies"]]
                all_studies.extend(filter(None, processed))
                pbar.update(len(processed))
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                time.sleep(self.rate_limit)
            except Exception as e:
                print(f"⚠️ Error: {e}")
                break
        pbar.close()
        df = pd.DataFrame(all_studies[:max_studies])
        if not df.empty:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            out_file = self.output_dir / f"raw_clinical_trials_{timestamp}.csv"
            df.to_csv(out_file, index=False)
            print(f"✅ {len(df)} trials saved to {out_file}")
        else:
            print("❌ No studies collected")
        return df

    def _extract_fields(self, study):
        try:
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            description = protocol.get("descriptionModule", {})
            status = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            collab_mod = protocol.get("sponsorCollaboratorsModule", {}) or {}
            lead_sponsor = collab_mod.get("leadSponsor", {}) or {}
            sponsor_name = lead_sponsor.get("name") or "unknown"
            sponsor_class = lead_sponsor.get("class") or "unknown"

            design_info = design.get("designInfo", {})
            masking_info = design_info.get("maskingInfo", {})
            enrollment_info = design.get("enrollmentInfo", {})
            arms_module = protocol.get("armsInterventionsModule", {})
            interventions_list = []
            interventions = arms_module.get("interventions", [])
            for intervention in interventions:
                interventions_list.append(
                    {
                        "name": intervention.get("name", ""),
                        "type": intervention.get("type", ""),
                        "description": intervention.get("description", ""),
                        "mesh_terms": intervention.get("meshTerms", []),
                        "other_ids": intervention.get("otherIds", []),
                    }
                )
            interventions_json = json.dumps(interventions_list, ensure_ascii=False)
            return {
                "sponsor": sponsor_name,
                "sponsor_class": sponsor_class,
                "nct_id": identification.get("nctId", ""),
                "brief_title": identification.get("briefTitle", ""),
                "official_title": identification.get("officialTitle", ""),
                "brief_summary": description.get("briefSummary", ""),
                "detailed_description": description.get("detailedDescription", ""),
                "overall_status": status.get("overallStatus", ""),
                "why_stopped": status.get("whyStopped", ""),
                "start_date": status.get("startDateStruct", {}).get("date", ""),
                "completion_date": status.get("completionDateStruct", {}).get(
                    "date", ""
                ),
                "primary_completion_date": status.get(
                    "primaryCompletionDateStruct", {}
                ).get("date", ""),
                "study_type": design.get("studyType", ""),
                "phases": "|".join(design.get("phases", []))
                if design.get("phases")
                else "",
                "allocation": design_info.get("allocation", ""),
                "intervention_model": design_info.get("interventionModel", ""),
                "masking": masking_info.get("masking", ""),
                "primary_purpose": design_info.get("primaryPurpose", ""),
                "enrollment_count": enrollment_info.get("count", 0),
                "enrollment_type": enrollment_info.get("type", ""),
                "interventions_json": interventions_json,
                "interventions_types": ";".join(
                    i.get("type", "") for i in interventions_list if i.get("type", "")
                ),
                "interventions_names": ";".join(
                    i.get("name", "") for i in interventions_list if i.get("name", "")
                ),
            }
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return None
