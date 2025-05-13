from MIMIC_IV_HAIM_API import *
import gc
from tqdm import tqdm

# General function that processes all data embeddings
def process_cxr_embeddings_haim_id(haim_id, dt_patient, df_init, df_imcxr, idx, df_stay_cxr):
    # DEMOGRAPHICS EMBEDDINGS EXTRACTION
    demo_embeddings = get_demographic_embeddings(dt_patient, verbose=0)
    gc.collect() #Clear memory
    
    # Time Series (TSFRESH-like) CHARTEVENT & LABEVENT EMBEDDINGS EXTRACTION
    aggregated_ts_ce_embeddings = get_ts_embeddings(dt_patient, event_type = 'chart')
    gc.collect() #Clear memory
    
    aggregated_ts_le_embeddings = get_ts_embeddings(dt_patient, event_type = 'lab')
    gc.collect() #Clear memory
    
    aggregated_ts_pe_embeddings = get_ts_embeddings(dt_patient, event_type = 'procedure')
    gc.collect() #Clear memory
    
    # CHEST XRAY VISION EMBEDDINGS EXTRACTION
    aggregated_densefeature_embeddings, _, aggregated_prediction_embeddings, _, _ = get_chest_xray_embeddings(dt_patient, verbose=0)
    gc.collect() #Clear memory
    
    # # NOTES FROM ECGs
    # aggregated_ecg_embeddings = get_notes_biobert_embeddings(patient, note_type = 'ecgnotes')
    # gc.collect() #Clear memory
    
    # # NOTES FROM ECOCARDIOGRAMs
    # aggregated_echo_embeddings = get_notes_biobert_embeddings(patient, note_type = 'echonotes')
    # gc.collect() #Clear memory
    
    # # NOTES FROM RADIOLOGY
    # aggregated_rad_embeddings = get_notes_biobert_embeddings(patient, note_type = 'radnotes')
    # gc.collect() #Clear memory

    # CHEST XRAY VISION SINGLE-IMAGE EMBEDDINGS EXTRACTION
    # print('getting xray')
    img = df_imcxr[idx]
    densefeature_embeddings, prediction_embeddings = get_single_chest_xray_embeddings(img)
    gc.collect() #Clear memory

    # Create Dataframes filteed by ordered sample number for Fusion
    df_haim_ids_fusion = pd.DataFrame([haim_id],columns=['haim_id'])
    df_demographics_embeddings_fusion = pd.DataFrame(demo_embeddings.reshape(1,-1), columns=['de_'+str(i) for i in range(demo_embeddings.shape[0])])
    df_ts_ce_embeddings_fusion = pd.DataFrame(aggregated_ts_ce_embeddings.values.reshape(1,-1), columns=['ts_ce_'+str(i) for i in range(aggregated_ts_ce_embeddings.values.shape[0])])
    df_ts_le_embeddings_fusion = pd.DataFrame(aggregated_ts_le_embeddings.values.reshape(1,-1), columns=['ts_le_'+str(i) for i in range(aggregated_ts_le_embeddings.values.shape[0])])
    df_ts_pe_embeddings_fusion = pd.DataFrame(aggregated_ts_pe_embeddings.values.reshape(1,-1), columns=['ts_pe_'+str(i) for i in range(aggregated_ts_pe_embeddings.values.shape[0])])
    
    df_vision_dense_embeddings_fusion = pd.DataFrame(densefeature_embeddings.reshape(1,-1), columns=['vd_'+str(i) for i in range(densefeature_embeddings.shape[0])])
    df_vision_predictions_embeddings_fusion = pd.DataFrame(prediction_embeddings.reshape(1,-1), columns=['vp_'+str(i) for i in range(prediction_embeddings.shape[0])])
    df_vision_multi_dense_embeddings_fusion = pd.DataFrame(aggregated_densefeature_embeddings.reshape(1,-1), columns=['vmd_'+str(i) for i in range(aggregated_densefeature_embeddings.shape[0])])
    df_vision_multi_predictions_embeddings_fusion = pd.DataFrame(aggregated_prediction_embeddings.reshape(1,-1), columns=['vmp_'+str(i) for i in range(aggregated_prediction_embeddings.shape[0])])
    # df_ecgnotes_embeddings_fusion = pd.DataFrame(aggregated_ecg_embeddings.reshape(1,-1), columns=['n_ecg_'+str(i) for i in range(aggregated_ecg_embeddings.shape[0])])
    # df_echonotes_embeddings_fusion = pd.DataFrame(aggregated_echo_embeddings.reshape(1,-1), columns=['n_ech_'+str(i) for i in range(aggregated_echo_embeddings.shape[0])])
    # df_radnotes_embeddings_fusion = pd.DataFrame(aggregated_rad_embeddings.reshape(1,-1), columns=['n_rad_'+str(i) for i in range(aggregated_rad_embeddings.shape[0])])
    
    # Vision targets
    cxr_target_columns = ['split','Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices', 'PerformedProcedureStepDescription','ViewPosition']
    df_vision_targets_fusion = df_stay_cxr.loc[idx:idx][cxr_target_columns].reset_index(drop=True)

    # Embeddings FUSION
    df_fusion = df_haim_ids_fusion
    df_fusion = pd.concat([df_fusion, df_init], axis=1)
    df_fusion = pd.concat([df_fusion, df_demographics_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_vision_dense_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_vision_predictions_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_vision_multi_dense_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_vision_multi_predictions_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_ts_ce_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_ts_le_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_ts_pe_embeddings_fusion], axis=1)
    # df_fusion = pd.concat([df_fusion, df_ecgnotes_embeddings_fusion], axis=1)
    # df_fusion = pd.concat([df_fusion, df_echonotes_embeddings_fusion], axis=1)
    # df_fusion = pd.concat([df_fusion, df_radnotes_embeddings_fusion], axis=1)
    
    #Add targets
    df_fusion = pd.concat([df_fusion, df_vision_targets_fusion], axis=1)
    gc.collect() #Clear memory
    
    return df_fusion


# Let's select a single HAIM Patient from pickle files and check if it fits inclusion criteria
def create_embedding_file(haim_id, file_path, fname):
    # Load precomputed file
    patient = load_patient_object(file_path)

    # Get information of chest x-rays conducted within this patiewnt stay
    df_cxr = patient.cxr
    df_imcxr = patient.imcxr
    admittime = patient.admissions.admittime.values[0]
    dischtime = patient.admissions.dischtime.values[0]
    df_stay_cxr = df_cxr.loc[(df_cxr['charttime'] >= admittime) & (df_cxr['charttime'] <= dischtime)]

    if not df_stay_cxr.empty:
        for idx, df_stay_cxr_row in df_stay_cxr.iterrows():
            # Get stay anchor times
            img_charttime = df_stay_cxr_row['charttime']
            img_deltacharttime = df_stay_cxr_row['deltacharttime']

            # Get time to discharge and discharge location/status
            img_id = df_stay_cxr_row["dicom_id"]
            img_length_of_stay = date_diff_hrs(dischtime, img_charttime)
            discharge_location = patient.core['discharge_location'][0]
            if discharge_location == "DIED": death_status = 1
            else: death_status = 0
                
            # Select allowed timestamp range
            start_hr = None
            end_hr = img_deltacharttime
            
            # We need to reload it since the original object has been modified
            patient = load_patient_object(file_path)
            dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)
            is_included = True

            # Convert to int 'gender_int', 'ethnicity_int', 'marital_status_int', 'language_int', 'insurance_int'
            int_mapping = {
                "gender": {"M": 1, "F": 0},
                "language": {
                    "English": 0,
                    "Spanish": 1,
                    "Russian": 2,
                    "Chinese": 3,
                    "Other": 4
                },
                "race": {
                    "White": 0,
                    "Black": 1,
                    "Hispanic": 2,
                    "Asian": 3,
                    "Other": 4
                },
                "marital_status": {
                    "MARRIED": 0,
                    "SINGLE": 1,
                    "WIDOWED": 2,
                    "DIVORCED": 3
                },
                "insurance": {
                    "Medicare": 0,
                    "Medicaid": 1,
                    "Private": 2,
                    "No charge": 3,
                    "Other": 4
                }
            }
            dt_patient.core["gender_int"] = dt_patient.core["gender"].map(int_mapping["gender"])
            dt_patient.core["marital_status_int"] = dt_patient.core["marital_status"].map(int_mapping["marital_status"])
            dt_patient.core["insurance_int"] = dt_patient.core["insurance"].map(int_mapping["insurance"])
            dt_patient.core["language_int"] = dt_patient.core["language"].apply(lambda x: int_mapping["language"].get(x, 4))
            dt_patient.core["ethnicity_int"] = dt_patient.core["race"].apply(lambda x: [v if k.lower() in x.lower() else 4 for k, v in int_mapping["race"].items()][0])
                
            if is_included:
                df_init = pd.DataFrame([[img_id, img_charttime, img_deltacharttime, discharge_location, img_length_of_stay, death_status]],columns=['img_id', 'img_charttime', 'img_deltacharttime', 'discharge_location', 'img_length_of_stay', 'death_status'])
                df_fusion = process_cxr_embeddings_haim_id(haim_id, dt_patient, df_init, df_imcxr, idx, df_stay_cxr)
                
                if os.path.isfile(fname):
                    df_fusion.to_csv(fname, mode='a', index=False, header=False)
                else:
                    os.makedirs(os.path.dirname(fname), exist_ok=True)
                    df_fusion.to_csv(fname, mode='w', index=False)



if __name__ == "__main__":
    # Run the code
    # Get all HAIM IDs from pickle files
    pickle_folder = 'data/haim_mimiciv/pickle/'
    pickle_files = sorted([f for f in os.listdir(pickle_folder) if f.endswith('.pkl')])
    haim_ids = [int(f.split('.')[0]) for f in pickle_files]

    # Create the output folder if it doesn't exist
    for haim_id in tqdm(haim_ids):
        file_path = pickle_folder + f"{haim_id:08d}" + '.pkl'
        fname = 'data/haim_mimiciv/embedding/' + f"{haim_id:08d}" + '.pkl'
        if not os.path.isfile(fname):
            create_embedding_file(haim_id, file_path, fname)