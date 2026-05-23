import anthropic
import pandas as pd

client = anthropic.Anthropic(api_key="X")

# =========================
# 🔧 JOUW PROMPT
# =========================
INSTRUCTION = """Bepaal de stance tegenover het GELUID van het Driftfestival.

Stances:
- Negatief: duidelijke hinder, slaapverstoring, frustratie
- Neutraal: lichte hinder maar acceptabel
- Positief: weinig tot geen hinder
- Geen stance: geen expliciete of impliciete evaluatie van geluid, ondanks aanwezigheid
- Niet blootgesteld: respondent vluchtte vooraf weg voor het geluid

Baseer de stance primair op Q15 (ervaren impact).
Gebruik Q21 ter ondersteuning of nuance, of om verborgen hinder te ontdekken wanneer Q15 vaag of minimaal is.
Gebruik ondersteunend:
- Q11, Q13, Q14, Q16, Q17
Negeer:
- irrelevante info
- verwachtingen (Q9), tenzij ervaring expliciet vermeld wordt
Bij twijfel:
-Q8, Q10
Bij twijfel tussen "Geen stance" en een andere stance:
→ kies NIET "Geen stance" als er enige evaluatie aanwezig is.
Kalibratie:
- Wees streng in het detecteren van hinder.
- Indien er duidelijke hinder of frustratie wordt vermeld (expliciet of impliciet), kies eerder "Negatief" dan "Neutraal".
- "Neutraal" is enkel voor lichte hinder zonder duidelijke frustratie of impact.
Prioriteit:
- Indien respondent niet aanwezig was tijdens het festival → Niet blootgesteld,
  zelfs als er sterke negatieve verwachtingen of ervaringen uit het verleden worden vermeld.
Cross-check:
- Baseer je niet uitsluitend op Q15.
- Indien andere antwoorden duidelijk hinder aangeven, mogen deze zwaarder doorwegen dan een vage of minimale Q15.
Negeer klachten over andere locaties of evenementen,
tenzij expliciet gebruikt om het geluid van Drift te evalueren.

Beslisproces:
1. Is er sprake van geluidshinder?
2. Hoe sterk is de hinder? (licht / matig / sterk)
3. Zijn er concrete gevolgen (bv. slaapverstoring)?
4. Bepaal stance:
   - Emotionele of versterkende taal (“heel erg”, “ondraaglijk”, “niet te doen”) → Negatief
   - Elke vorm van slaapverstoring → Negatief
   - Elke duidelijke frustratie of klacht → Negatief
   - Gedrag dat wijst op aanpassing (ramen sluiten, verplaatsen, oordoppen) → meestal Negatief
    -Lichte hinder zonder frustratie of klacht → Neutraal
   - Nauwelijks hinder → Positief
   - Geen evaluatie, ondanks aanwezigheid → Geen stance
   - Niet aanwezig tijdens het festival → Niet blootgesteld
5. Impliciet gedrag:
   - Gedrag zoals ramen sluiten, geluid horen maar tolereren → lichte hinder (Neutraal)
   - Gedrag zoals niet kunnen slapen, elders slapen → sterke hinder (Negatief)
   - Indien vooraf weggegaan en niet aanwezig → Niet blootgesteld

Context:
Respondenten wonen in een stedelijke omgeving met bestaande achtergrondgeluiden en andere evenementen.
Vergelijkingen met andere evenementen kunnen relevant zijn voor het inschatten van hinder.

De onderstaande voorbeelden tonen hoe de stance bepaald wordt op basis van relevante antwoorden. Focus op Q15 en Q21.
Voorbeeld 3243:
- Q15: Overdag weinig last, maar 's avonds en vooral zondagavond duidelijke hinder. Nachtrust van kinderen verstoord.
- Q21: Festivals overdag organiseren en einduur vroeger leggen. 
- STANCE: Negatief
- Reden: Duidelijke overlast. Nachtrust verstoord.

Voorbeeld 3344:
- Q15: We zijn preventief buitenshuis gaan overnachten. 
- Q21: Akoestische maatregelen, tijdelijk of permanent. 
- STANCE: Niet blootgesteld
- Reden: Niet aanwezig tijdens festival, dus geen directe evaluatie van geluid

Voorbeeld 3320:
- Q15: Deze keer geen invloed
- Q21: Vooral dj's zijn storend en het zou max tot 1 u 's nachts mogen duren. 4 u is veel te lang. Kortrijk weide en zeker onder die brug is niet toelaatbaar, Nelson Mandelaplein geeft geen of weinig overlast
- STANCE: Positief
- Reden: Geen hinder

Voorbeeld 3257:
- Q15: Niet echt last !
- Q21: Geen evenementen meer op Kortrijk weide , bv met sinksenfeesten! Dj onder de brug, dat is niet te doen !
- STANCE: Positief
- Reden: Geen hinder

Voorbeeld 3232:
- Q15: Als je wilt slapen blijft het gedreun, zo omschrijf ik het " muziek" niet mijn genre vooral
- Q21: Lager decibels!!!
- STANCE: Negatief
- Reden: Nachtrust verstoord

Voorbeeld 3415:
- Q15: Mijn slaapkamer is gericht op het festivalterrein, dus kan ik pas later op de avond in bed gaan slapen. Daarvoor slaap ik in de woonkamer.
- Q21: De festivals trekken veel volk. Zolang er niet elk weekend één plaatsvindt, en de regelgeving gerespecteerd worden, heb ik er niet echt een probleem mee.
- STANCE: Neutraal
- Reden: Enige hinder, maar zonder overlast. Geen zware emotionele taal

Voorbeeld 3462:
- Q15: Raam kon niet open blijven
- Q21: /
- STANCE: Neutraal
- Reden: Enige hinder, maar zonder overlast. Geen zware emotionele taal

Voorbeeld 3251:
- Q15: Heel erg
- Q21: Dergelijke evenementen vroeger te laten stoppen. Nachtrust niet storen. Zeker NOOIT op zondagen en weekdagen
- STANCE: Negatief
- Reden: Negatieve emoties, versterkende woorden, duidelijke hinder

Voorbeeld 3518:
- Q15: Geen
- Q21: Geen
- STANCE: Geen stance
- Reden: Geen evaluatie

Voorbeeld 3535:
- Q15: Zeer weinig. Indien ik het festival wilde horen, zette ik de ramen open. Indien niet en wanneer ik ging slapen, deed ik ze dicht. Dan was er amper iets te horen.
- Q21: Een eventuele korting voor dichte buurtbewoners om naar het festival te gaan.
- STANCE: Positief
- Reden: Geen hinder

Geef output exact in dit formaat:
STANCE: <Negatief / Neutraal / Positief / Geen stance / Niet blootgesteld>
REDEN: <max 50 woorden, focus op hinder en impact>
"""

# =========================
# 🔧 CSV INLEZEN
# =========================
file_path = r"C:\Users\Gebruiker\Documents\DATA\Stance_Detection_LLM\SD_Geluid_promptversie.csv"

df = pd.read_csv(file_path, sep=";", encoding="latin1")

# =========================
# 🔍 KOLOMMEN AUTOMATISCH VINDEN
# =========================
id_col = None
q15_col = None
q21_col = None
q9_col = None
q16_col = None
q8_col = None
q10_col = None

for col in df.columns:
    if "visitor_id" in col.lower() or col.lower() == "VRAAG":
        id_col = col
    if "invloed" in col.lower() or "nachtrust" in col.lower():
        q15_col = col
    if "ideale oplossing" in col.lower():
        q21_col = col
    if "voorafgaand" in col.lower():
        q9_col = col
    if "geluidsbronnen" in col.lower():
        q16_col = col
    if "algemene ervaring" in col.lower():
        q8_col = col
    if "hoeverre hinderde" in col.lower():
        q10_col = col


print("Q15 kolom:", q15_col)
print("Q21 kolom:", q21_col)
print("Q9 kolom:", q9_col)
print("Q16 kolom:", q16_col)
print("Q8 kolom:", q8_col)
print("Q10 kolom:", q10_col)
print("ID kolom:", id_col)

# =========================
# 🧠 CLASSIFICATIE FUNCTIE
# =========================
def classify(q15, q21, q9, q16, q8, q10):
    user_input = f"""
    Q15: {q15}
    Q21: {q21}
    Q9: {q9}
    Q16: {q16}
    Q8: {q8}
    Q10: {q10}
    """

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0,
        system=INSTRUCTION,
        messages=[{"role": "user", "content": user_input}]
    )

    return response.content[0].text


# =========================
# 🧪 TEST 1 CASE (BELANGRIJK)
# =========================
test_index = 23  # ← pas aan indien nodig

print("\n--- TEST CASE ---")

test_q15 = str(df.iloc[test_index][q15_col])
test_q21 = str(df.iloc[test_index][q21_col])
test_q9 = str(df.iloc[test_index][q9_col])
test_q16 = str(df.iloc[test_index][q16_col])
test_q8 = str(df.iloc[test_index][q8_col])
test_q10 = str(df.iloc[test_index][q10_col])

result = classify(test_q15, test_q21, test_q9, test_q16, test_q8, test_q10)

print("Input Q15:", test_q15)
print("Input Q21:", test_q21)
print("Input Q9:", test_q9)
print("Input Q16:", test_q16)
print("Input Q8:", test_q8)
print("Input Q10:", test_q10)
print("Output:\n", result)

test_only = False

if test_only:
    print("\nTestmodus actief → script stopt na 1 case.")
    exit()

# =========================
# ❓ STOP HIER VOOR TEST
# =========================
run_full = input("\nWil je de volledige dataset runnen? (y/n): ")

if run_full.lower() != "y":
    print("Gestopt na test.")
    exit()


# =========================
# 🔄 VOLLEDIGE DATASET
# =========================
stances = []
redenen = []

for index, row in df.iterrows():
    q15 = str(row[q15_col])
    q21 = str(row[q21_col])
    q9 = str(row[q9_col])
    q16 = str(row[q16_col])
    q8 = str(row[q8_col])
    q10 = str(row[q10_col])

    try:
        output = classify(q15, q21, q9, q16, q8, q10)

        # simpele parsing
        stance = ""
        reden = ""

        if "STANCE:" in output:
            stance = output.split("STANCE:")[1].split("\n")[0].strip()
        if "REDEN:" in output:
            reden = output.split("REDEN:")[1].strip()

    except Exception as e:
        print(f"Fout bij rij {index}: {e}")
        stance = "ERROR"
        reden = "ERROR"

    stances.append(stance)
    redenen.append(reden)

    print(f"Rij {index} verwerkt")

# =========================
# 💾 OPSLAAN NIEUWE CSV
# =========================
df["STANCE_GPT"] = stances
df["REDEN_GPT"] = redenen

output_file = "output_stance3.csv"
df.to_csv(output_file, sep=";", index=False, encoding="latin1")

print("\nKlaar! Bestand opgeslagen als:", output_file)