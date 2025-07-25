import pytest

from recruitment_fairness.data.loader import ClinicalTrialsWebCollector


@pytest.fixture
def mock_study():
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT0001",
                "briefTitle": "Test Trial",
                "officialTitle": "Official Test Trial Title",
            },
            "descriptionModule": {
                "briefSummary": "Short summary of the trial.",
                "detailedDescription": "Detailed description here.",
            },
            "statusModule": {
                "overallStatus": "Completed",
                "whyStopped": "",
                "startDateStruct": {"date": "2020-01-01"},
                "completionDateStruct": {"date": "2021-01-01"},
                "primaryCompletionDateStruct": {"date": "2020-12-01"},
            },
            "designModule": {
                "studyType": "Interventional",
                "phases": ["Phase 2"],
                "designInfo": {
                    "allocation": "Randomized",
                    "interventionModel": "Parallel Assignment",
                    "maskingInfo": {"masking": "Double"},
                    "primaryPurpose": "Treatment",
                },
                "enrollmentInfo": {"count": 100, "type": "Actual"},
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Mock Sponsor", "class": "Industry"}
            },
            "armsInterventionsModule": {
                "interventions": [
                    {
                        "name": "Drug A",
                        "type": "Drug",
                        "description": "Test drug description",
                        "meshTerms": ["MESH1"],
                        "otherIds": [],
                    }
                ]
            },
        }
    }


def test_extract_fields_structure(mock_study):
    collector = ClinicalTrialsWebCollector()
    extracted = collector._extract_fields(mock_study)

    assert isinstance(extracted, dict)
    assert extracted["nct_id"] == "NCT0001"
    assert extracted["brief_title"] == "Test Trial"
    assert extracted["sponsor"] == "Mock Sponsor"
    assert extracted["study_type"] == "Interventional"
    assert extracted["phases"] == "Phase 2"
    assert "interventions_names" in extracted
