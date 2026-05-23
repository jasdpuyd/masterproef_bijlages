VERDUIDELIJKING VAN SCRIPTS

VERWERKING GELUIDSDATA
aggregatie_tijd.py			Aggregatie van geluidsdata per minuut, per 5 minuten en per 10 minuten	
aggregatie_festivalblokken.py		Aggregatie van geluidsdata per festivalblok
wav_conversie_dBA.ipynb			Omzetting van WAV-data naar dBA, via A-weging en kalibratie-offset
verlopen.py				Genereren van grafieken met dBA-verloop per meter
interpolatie.py				Genereren van rasterkaarten via IDW-interpolatie

KOPPELEN VAN DATASET
kopppeling_geluid_polygonen.py		Koppelen van shapefile (PPGIS), CSV (enquête) en berekening van geluidsmetrieken

VERKENNENDE ANALYSES
vergelijking_blokken.py			Vergelijkende statistische tests tussen festivalblokken
vergelijking_dagen.py			Vergelijkende statistische tests tussen dagen
moransi.py				Ruimtelijke autocorrelatietest
wetgeving_test.py			Controle van het naleven van de geluidsnorm

CORRELATIE & REGRESSIE
correllatie.py				Correlatietest
regressie_model1.py			Regressieanalyse - Model 1: Geluidsmetrieken
regressie_model2.py			Regressieanalyse - Model 2: Ruimtelijke factoren
regressie_model3.py			Regressieanalyse - Model 3: Volledig model
PCA_regressie.py			PCA-regressie

STANCE DETECTION
stance_claude.py			Prompt gebruikt voor stance detection via Claude Sonnet 4
stance_openai.py			Prompt gebruikt voor stance detection via GPT-4.1-mini
stance_mistral.py			Prompt gebruikt voor stance detection via Mistral-Large-2411

