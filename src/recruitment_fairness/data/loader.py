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

    def search_trials(self, query_term="", max_studies=10000):
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
                studies = data.get("studies", [])
                if not studies:
                    break

                processed = [self._extract_fields(s) for s in studies]
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
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            out = self.output_dir / f"raw_clinical_trials_{ts}.csv"
            df.to_csv(out, index=False)
            print(f"✅ {len(df)} trials saved to {out}")
        else:
            print("❌ No studies collected")
        return df

    def _extract_fields(self, study):
        try:
            proto = study.get("protocolSection", {}) or {}
            ident = proto.get("identificationModule", {}) or {}
            desc = proto.get("descriptionModule", {}) or {}
            status = proto.get("statusModule", {}) or {}
            design = proto.get("designModule", {}) or {}

            # Eligibility
            elig = proto.get("eligibilityModule", {}) or {}
            eligibility_criteria = ""
            crit = elig.get("eligibilityCriteria") or elig.get("criteria")
            if isinstance(crit, dict):
                eligibility_criteria = (
                    crit.get("textBlock") or crit.get("textblock") or ""
                )
            elif isinstance(crit, str):
                eligibility_criteria = crit

            # Sponsor
            collab = proto.get("sponsorCollaboratorsModule", {}) or {}
            lead = collab.get("leadSponsor", {}) or {}
            sponsor = lead.get("name", "unknown")
            sponsor_class = lead.get("class", "OTHER")

            # Enrollment
            enroll = design.get("enrollmentInfo", {}) or {}
            e_type = (enroll.get("type") or "").lower()
            e_cnt = enroll.get("count", None)
            planned_enrollment = e_cnt if "anticipated" in e_type else None
            actual_enrollment = e_cnt if "actual" in e_type else None
            planned_enrollment = planned_enrollment or e_cnt
            actual_enrollment = actual_enrollment or e_cnt

            def to_ts(x):
                try:
                    return pd.to_datetime(x)
                except (ValueError, TypeError):
                    return pd.NaT

            start_str = status.get("startDateStruct", {}).get("date", "")
            prim_str = status.get("primaryCompletionDateStruct", {}).get("date", "")
            comp_str = status.get("completionDateStruct", {}).get("date", "")

            start_ts = to_ts(start_str)
            prim_ts = to_ts(prim_str)
            comp_ts = to_ts(comp_str)

            planned_duration_m = (
                ((prim_ts - start_ts).days / 30.0)
                if not pd.isna(start_ts) and not pd.isna(prim_ts)
                else None
            )
            actual_duration_m = (
                ((comp_ts - start_ts).days / 30.0)
                if not pd.isna(start_ts) and not pd.isna(comp_ts)
                else None
            )

            # Arm count
            arms_mod = proto.get("armsInterventionsModule", {}) or {}
            num_arms = len(arms_mod.get("armGroups", []))

            # DMC
            oversight = proto.get("oversightModule", {}) or {}
            has_dmc = int(bool(oversight.get("oversightHasDmc", False)))

            # Multi-country
            contacts_mod = proto.get("contactsLocationsModule", {}) or {}
            contacts = contacts_mod.get("locations", []) or []
            multi_country = int(len(contacts) > 1)

            country = (
                contacts[0].get("country", "").strip()
                if contacts and isinstance(contacts[0], dict)
                else ""
            )

            conditions_mod = proto.get("conditionsModule", {}) or {}
            conditions_list = conditions_mod.get("conditions", [])
            condition_str = conditions_list[0] if conditions_list else ""

            # Interventions
            interventions = []
            for iv in arms_mod.get("interventions", []):
                interventions.append(
                    {
                        "name": iv.get("name", ""),
                        "type": iv.get("type", ""),
                        "description": iv.get("description", ""),
                        "mesh_terms": iv.get("meshTerms", []),
                        "other_ids": iv.get("otherIds", []),
                    }
                )
            iv_json = json.dumps(interventions, ensure_ascii=False)

            raw_phases = design.get("phases") or []
            if isinstance(raw_phases, str):
                phases_str = raw_phases
            else:
                phases_str = "|".join(raw_phases)

            return {
                "sponsor": sponsor,
                "sponsor_class": sponsor_class,
                "nct_id": ident.get("nctId", ""),
                "brief_title": ident.get("briefTitle", ""),
                "official_title": ident.get("officialTitle", ""),
                "brief_summary": desc.get("briefSummary", ""),
                "detailed_description": desc.get("detailedDescription", ""),
                "eligibility_criteria": eligibility_criteria,
                "overall_status": status.get("overallStatus", ""),
                "why_stopped": status.get("whyStopped", ""),
                "start_date": start_str,
                "primary_completion_date": prim_str,
                "completion_date": comp_str,
                "study_type": design.get("studyType", ""),
                "phases": phases_str,
                "allocation": design.get("designInfo", {}).get("allocation", ""),
                "intervention_model": design.get("designInfo", {}).get(
                    "interventionModel", ""
                ),
                "masking": design.get("designInfo", {}).get("masking", ""),
                "primary_purpose": design.get("designInfo", {}).get(
                    "primaryPurpose", ""
                ),
                "planned_enrollment": planned_enrollment,
                "actual_enrollment": actual_enrollment,
                "planned_duration_m": planned_duration_m,
                "actual_duration_m": actual_duration_m,
                "num_arms": num_arms,
                "has_dmc": has_dmc,
                "multi_country": multi_country,
                "first_country": country,
                "condition": condition_str,
                "enrollment_count": enroll.get("count", 0),
                "enrollment_type": enroll.get("type", ""),
                "interventions_json": iv_json,
                "interventions_types": ";".join(
                    i["type"] for i in interventions if i["type"]
                ),
                "interventions_names": ";".join(
                    i["name"] for i in interventions if i["name"]
                ),
            }

        except Exception as e:
            ident = (
                study.get("protocolSection", {})
                .get("identificationModule", {})
                .get("nctId", "UNKNOWN")
            )
            print(f"❌ Extraction error for NCT {ident}: {e}")
            return None
