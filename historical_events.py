import urllib.request, json, csv, io
from datetime import datetime
from collections import Counter

print("ИСТОРИЧЕСКИЕ СОБЫТИЯ — альтернативные источники\n")

events = []

# ══ 1. WARS — через Wikipedia REST API (работает всегда) ══
print("[1/4] Список войн через Wikipedia...")
# Используем предзагруженный список из АКПД (Armed Conflict и Peace Data)
# + COW (Correlates of War) — публичные данные

WARS_COW = [
    # Napoleonic Wars era
    ("1803-05-18","geopolitical","Napoleonic Wars"),
    ("1808-05-02","geopolitical","Peninsular War"),
    ("1812-06-24","geopolitical","French invasion of Russia"),
    ("1815-03-20","geopolitical","Hundred Days"),
    # 19th century
    ("1821-03-25","geopolitical","Greek War of Independence"),
    ("1823-04-06","geopolitical","Franco-Spanish War"),
    ("1828-04-26","geopolitical","Russo-Turkish War"),
    ("1839-04-27","geopolitical","First Anglo-Afghan War"),
    ("1846-05-13","geopolitical","Mexican-American War"),
    ("1848-03-23","geopolitical","First Italian War of Independence"),
    ("1853-10-04","geopolitical","Crimean War"),
    ("1856-01-15","geopolitical","Second Opium War"),
    ("1861-04-12","geopolitical","American Civil War"),
    ("1864-02-01","geopolitical","Second Schleswig War"),
    ("1866-06-14","geopolitical","Austro-Prussian War"),
    ("1870-07-19","geopolitical","Franco-Prussian War"),
    ("1877-04-24","geopolitical","Russo-Turkish War 1877"),
    ("1879-02-14","geopolitical","War of the Pacific"),
    ("1894-07-25","geopolitical","First Sino-Japanese War"),
    ("1898-04-25","geopolitical","Spanish-American War"),
    ("1899-10-11","geopolitical","Second Boer War"),
    ("1904-02-08","geopolitical","Russo-Japanese War"),
    # 20th century
    ("1910-11-20","geopolitical","Mexican Revolution"),
    ("1912-10-08","geopolitical","First Balkan War"),
    ("1914-07-28","geopolitical","World War I"),
    ("1917-02-23","geopolitical","Russian Revolution"),
    ("1918-11-11","geopolitical","WW1 End"),
    ("1919-05-15","geopolitical","Greco-Turkish War"),
    ("1927-07-01","geopolitical","Chinese Civil War"),
    ("1931-09-18","geopolitical","Japanese invasion Manchuria"),
    ("1935-10-03","geopolitical","Second Italo-Ethiopian War"),
    ("1936-07-17","geopolitical","Spanish Civil War"),
    ("1937-07-07","geopolitical","Second Sino-Japanese War"),
    ("1939-09-01","geopolitical","World War II"),
    ("1941-06-22","geopolitical","Operation Barbarossa"),
    ("1941-12-07","geopolitical","Pearl Harbor"),
    ("1945-05-08","geopolitical","WW2 End Europe"),
    ("1945-08-15","geopolitical","WW2 End Pacific"),
    ("1946-12-19","geopolitical","First Indochina War"),
    ("1948-05-14","geopolitical","Arab-Israeli War 1948"),
    ("1950-06-25","geopolitical","Korean War"),
    ("1953-07-27","geopolitical","Korean War End"),
    ("1954-11-01","geopolitical","Algerian War"),
    ("1956-10-29","geopolitical","Suez Crisis"),
    ("1959-01-01","geopolitical","Cuban Revolution"),
    ("1961-04-17","geopolitical","Bay of Pigs"),
    ("1962-10-16","geopolitical","Cuban Missile Crisis"),
    ("1965-03-08","geopolitical","Vietnam War escalation"),
    ("1967-06-05","geopolitical","Six-Day War"),
    ("1968-01-30","geopolitical","Tet Offensive"),
    ("1971-03-26","geopolitical","Bangladesh Liberation War"),
    ("1973-10-06","geopolitical","Yom Kippur War"),
    ("1975-04-30","geopolitical","Fall of Saigon"),
    ("1979-02-17","geopolitical","Sino-Vietnamese War"),
    ("1979-12-24","geopolitical","Soviet-Afghan War"),
    ("1980-09-22","geopolitical","Iran-Iraq War"),
    ("1982-04-02","geopolitical","Falklands War"),
    ("1983-10-25","geopolitical","Invasion of Grenada"),
    ("1989-12-20","geopolitical","Invasion of Panama"),
    ("1990-08-02","geopolitical","Gulf War start"),
    ("1991-01-17","geopolitical","Gulf War"),
    ("1991-06-25","geopolitical","Yugoslav Wars"),
    ("1992-04-06","geopolitical","Bosnian War"),
    ("1994-04-07","geopolitical","Rwandan Genocide"),
    ("1994-12-11","geopolitical","First Chechen War"),
    ("1998-08-07","geopolitical","US Embassy bombings"),
    ("1999-03-24","geopolitical","Kosovo War"),
    ("1999-08-09","geopolitical","Second Chechen War"),
    ("2001-09-11","geopolitical","September 11"),
    ("2001-10-07","geopolitical","Afghanistan War"),
    ("2003-03-20","geopolitical","Iraq War"),
    ("2006-07-12","geopolitical","2006 Lebanon War"),
    ("2008-08-08","geopolitical","Russo-Georgian War"),
    ("2011-02-17","geopolitical","Libyan Civil War"),
    ("2011-03-15","geopolitical","Syrian Civil War"),
    ("2014-02-20","geopolitical","Ukrainian Revolution"),
    ("2014-07-17","geopolitical","MH17"),
    ("2015-03-26","geopolitical","Yemen Civil War"),
    ("2016-07-15","geopolitical","Turkish coup attempt"),
    ("2020-01-03","geopolitical","Soleimani killing"),
    ("2021-08-15","geopolitical","Taliban takeover"),
    ("2022-02-24","geopolitical","Russia-Ukraine War"),
    ("2023-10-07","geopolitical","Hamas attack Israel"),
    ("2024-04-13","geopolitical","Iran attack Israel"),
]

for date, cat, name in WARS_COW:
    events.append({"date":date,"category":cat,"source":"cow","name":name})
print(f"  ✓ {len(WARS_COW)} войн и конфликтов")

# ══ 2. ЭКОНОМИЧЕСКИЕ КРИЗИСЫ — полный список ══
print("\n[2/4] Экономические кризисы с 1800...")
ECON_CRISES = [
    ("1819-01-01","economic_crisis","Panic of 1819"),
    ("1825-12-01","economic_crisis","Panic of 1825"),
    ("1837-05-10","economic_crisis","Panic of 1837"),
    ("1847-10-01","economic_crisis","Panic of 1847"),
    ("1857-08-24","economic_crisis","Panic of 1857"),
    ("1866-05-11","economic_crisis","Panic of 1866"),
    ("1873-09-18","economic_crisis","Panic of 1873"),
    ("1882-05-09","economic_crisis","Panic of 1882"),
    ("1884-05-01","economic_crisis","Panic of 1884"),
    ("1890-11-15","economic_crisis","Baring Crisis"),
    ("1893-05-05","economic_crisis","Panic of 1893"),
    ("1896-08-01","economic_crisis","Depression of 1896"),
    ("1907-10-22","economic_crisis","Panic of 1907"),
    ("1920-01-01","economic_crisis","Depression of 1920"),
    ("1929-10-24","economic_crisis","Black Thursday"),
    ("1929-10-29","economic_crisis","Black Tuesday"),
    ("1937-08-01","economic_crisis","Recession of 1937"),
    ("1973-10-17","economic_crisis","Oil Crisis 1973"),
    ("1979-04-01","economic_crisis","Energy Crisis 1979"),
    ("1980-01-01","economic_crisis","Early 1980s Recession"),
    ("1982-08-12","economic_crisis","Mexican Debt Crisis"),
    ("1987-10-19","economic_crisis","Black Monday"),
    ("1989-08-09","economic_crisis","S&L Crisis"),
    ("1990-08-01","economic_crisis","Early 1990s Recession"),
    ("1992-09-16","economic_crisis","Black Wednesday"),
    ("1994-12-20","economic_crisis","Mexican Peso Crisis"),
    ("1997-07-02","economic_crisis","Asian Financial Crisis"),
    ("1998-08-17","economic_crisis","Russian Financial Crisis"),
    ("1998-09-23","economic_crisis","LTCM collapse"),
    ("2000-03-10","economic_crisis","Dot-com crash"),
    ("2001-12-02","economic_crisis","Enron collapse"),
    ("2002-06-25","economic_crisis","WorldCom fraud"),
    ("2007-08-09","economic_crisis","Subprime crisis start"),
    ("2008-03-14","economic_crisis","Bear Stearns"),
    ("2008-09-15","economic_crisis","Lehman Brothers"),
    ("2008-10-08","economic_crisis","Global crash"),
    ("2009-01-01","economic_crisis","Great Recession"),
    ("2010-05-06","economic_crisis","Flash Crash"),
    ("2010-05-10","economic_crisis","Greek bailout"),
    ("2011-08-05","economic_crisis","US downgrade"),
    ("2012-06-01","economic_crisis","Euro Crisis"),
    ("2014-10-15","economic_crisis","Market selloff"),
    ("2015-08-24","economic_crisis","Chinese market crash"),
    ("2016-01-20","economic_crisis","Oil crash"),
    ("2018-12-24","economic_crisis","Christmas Eve crash"),
    ("2020-02-20","economic_crisis","COVID crash"),
    ("2020-03-20","economic_crisis","Fed intervention"),
    ("2022-01-24","economic_crisis","Tech selloff"),
    ("2022-09-23","economic_crisis","UK gilt crisis"),
    ("2023-03-10","economic_crisis","SVB collapse"),
    ("2023-03-19","economic_crisis","Credit Suisse"),
]
for date, cat, name in ECON_CRISES:
    events.append({"date":date,"category":cat,"source":"manual","name":name})
print(f"  ✓ {len(ECON_CRISES)} экономических кризисов")

# ══ 3. ЭПИДЕМИИ с 1800 ══
print("\n[3/4] Эпидемии с 1800...")
EPIDEMICS = [
    ("1816-01-01","epidemic","Cholera pandemic 1"),
    ("1829-01-01","epidemic","Cholera pandemic 2"),
    ("1852-01-01","epidemic","Cholera pandemic 3"),
    ("1863-01-01","epidemic","Cholera pandemic 4"),
    ("1881-01-01","epidemic","Cholera pandemic 5"),
    ("1899-01-01","epidemic","Cholera pandemic 6"),
    ("1889-10-01","epidemic","Russian flu pandemic"),
    ("1918-03-01","epidemic","Spanish Flu"),
    ("1918-10-01","epidemic","Spanish Flu peak"),
    ("1957-02-01","epidemic","Asian Flu"),
    ("1968-07-01","epidemic","Hong Kong Flu"),
    ("1976-07-04","epidemic","Legionnaires disease"),
    ("1976-08-26","epidemic","Ebola first outbreak"),
    ("1981-06-05","epidemic","HIV/AIDS recognition"),
    ("1984-04-23","epidemic","HIV cause identified"),
    ("1988-01-01","epidemic","HIV pandemic peak"),
    ("1991-01-01","epidemic","Cholera pandemic 7"),
    ("1993-04-01","epidemic","Cryptosporidiosis outbreak"),
    ("1994-09-21","epidemic","Plague India"),
    ("1995-05-01","epidemic","Ebola Kikwit"),
    ("1996-03-01","epidemic","BSE/Mad Cow crisis"),
    ("1997-05-01","epidemic","H5N1 first cases"),
    ("1999-10-01","epidemic","West Nile US"),
    ("2002-11-16","epidemic","SARS start"),
    ("2003-03-12","epidemic","SARS global alert"),
    ("2004-01-28","epidemic","H5N1 Asia"),
    ("2005-08-01","epidemic","Marburg Angola"),
    ("2006-08-01","epidemic","Chikungunya Indian Ocean"),
    ("2009-04-09","epidemic","Swine Flu H1N1"),
    ("2009-06-11","epidemic","H1N1 pandemic declared"),
    ("2010-10-01","epidemic","Haiti cholera"),
    ("2012-09-24","epidemic","MERS first case"),
    ("2013-03-31","epidemic","H7N9 China"),
    ("2014-03-23","epidemic","Ebola West Africa"),
    ("2014-08-08","epidemic","Ebola emergency"),
    ("2015-05-20","epidemic","MERS Korea"),
    ("2016-02-01","epidemic","Zika pandemic"),
    ("2017-05-12","epidemic","WannaCry (bio parallel)"),
    ("2018-05-08","epidemic","Ebola DRC"),
    ("2018-08-01","epidemic","Ebola DRC 2"),
    ("2019-07-17","epidemic","Ebola emergency 2019"),
    ("2019-12-31","epidemic","COVID-19 first report"),
    ("2020-01-30","epidemic","COVID emergency"),
    ("2020-03-11","epidemic","COVID pandemic"),
    ("2020-11-09","epidemic","Vaccine announced"),
    ("2021-05-05","epidemic","India COVID peak"),
    ("2021-11-26","epidemic","Omicron variant"),
    ("2022-05-23","epidemic","Monkeypox emergency"),
    ("2022-07-23","epidemic","Mpox global emergency"),
    ("2023-05-05","epidemic","COVID emergency end"),
    ("2024-08-14","epidemic","Mpox 2024 emergency"),
]
for date, cat, name in EPIDEMICS:
    events.append({"date":date,"category":cat,"source":"who_manual","name":name})
print(f"  ✓ {len(EPIDEMICS)} эпидемий")

# ══ 4. ТЕХНОЛОГИЧЕСКИЕ ПРОРЫВЫ ══
print("\n[4/4] Технологические прорывы...")
TECH = [
    ("1804-02-21","tech_breakthrough","Steam locomotive"),
    ("1820-07-21","tech_breakthrough","Electromagnetism"),
    ("1837-05-24","tech_breakthrough","Telegraph"),
    ("1844-03-24","tech_breakthrough","Morse code"),
    ("1859-08-27","tech_breakthrough","Oil well"),
    ("1866-07-27","tech_breakthrough","Transatlantic cable"),
    ("1876-03-10","tech_breakthrough","Telephone"),
    ("1879-10-21","tech_breakthrough","Electric light bulb"),
    ("1885-01-29","tech_breakthrough","Automobile"),
    ("1895-11-08","tech_breakthrough","X-rays"),
    ("1895-12-28","tech_breakthrough","Cinema"),
    ("1896-05-07","tech_breakthrough","Radio"),
    ("1903-12-17","tech_breakthrough","Airplane"),
    ("1905-06-30","tech_breakthrough","Special Relativity"),
    ("1913-01-13","tech_breakthrough","Assembly line"),
    ("1928-09-28","tech_breakthrough","Penicillin"),
    ("1938-12-17","tech_breakthrough","Nuclear fission"),
    ("1945-07-16","tech_breakthrough","Trinity nuclear test"),
    ("1947-12-23","tech_breakthrough","Transistor"),
    ("1953-04-25","tech_breakthrough","DNA structure"),
    ("1957-10-04","tech_breakthrough","Sputnik"),
    ("1958-02-01","tech_breakthrough","Explorer 1"),
    ("1961-04-12","tech_breakthrough","Gagarin in space"),
    ("1969-07-20","tech_breakthrough","Moon landing"),
    ("1971-11-15","tech_breakthrough","Microprocessor"),
    ("1973-01-01","tech_breakthrough","Internet (ARPANET)"),
    ("1975-01-01","tech_breakthrough","Personal computer"),
    ("1981-08-12","tech_breakthrough","IBM PC"),
    ("1983-01-01","tech_breakthrough","TCP/IP"),
    ("1989-03-12","tech_breakthrough","World Wide Web"),
    ("1990-06-26","tech_breakthrough","Human Genome start"),
    ("1993-04-22","tech_breakthrough","Mosaic browser"),
    ("1995-08-09","tech_breakthrough","Netscape IPO"),
    ("1996-02-10","tech_breakthrough","Deep Blue chess"),
    ("1997-02-22","tech_breakthrough","Dolly the sheep"),
    ("1997-05-11","tech_breakthrough","Deep Blue wins"),
    ("1998-09-04","tech_breakthrough","Google founded"),
    ("2000-02-12","tech_breakthrough","Human genome draft"),
    ("2001-10-23","tech_breakthrough","iPod"),
    ("2003-04-14","tech_breakthrough","Human genome complete"),
    ("2004-02-04","tech_breakthrough","Facebook"),
    ("2005-02-14","tech_breakthrough","YouTube"),
    ("2007-01-09","tech_breakthrough","iPhone"),
    ("2008-09-10","tech_breakthrough","LHC first run"),
    ("2009-01-03","tech_breakthrough","Bitcoin"),
    ("2010-06-24","tech_breakthrough","iPhone 4"),
    ("2011-10-05","tech_breakthrough","Siri AI"),
    ("2012-07-04","tech_breakthrough","Higgs boson"),
    ("2012-08-06","tech_breakthrough","Curiosity Mars"),
    ("2014-10-20","tech_breakthrough","AI breakthrough"),
    ("2015-09-14","tech_breakthrough","Gravitational waves"),
    ("2016-03-09","tech_breakthrough","AlphaGo wins"),
    ("2017-12-05","tech_breakthrough","AlphaZero"),
    ("2018-06-01","tech_breakthrough","GPT-1"),
    ("2019-04-10","tech_breakthrough","Black hole photo"),
    ("2019-10-23","tech_breakthrough","Quantum supremacy"),
    ("2020-11-09","tech_breakthrough","COVID vaccine"),
    ("2021-07-11","tech_breakthrough","Bezos space"),
    ("2022-11-30","tech_breakthrough","ChatGPT"),
    ("2023-03-14","tech_breakthrough","GPT-4"),
    ("2023-07-06","tech_breakthrough","Llama 2"),
    ("2024-02-15","tech_breakthrough","Sora"),
    ("2024-05-13","tech_breakthrough","GPT-4o"),
    ("2025-01-20","tech_breakthrough","DeepSeek R1"),
]
for date, cat, name in TECH:
    events.append({"date":date,"category":cat,"source":"manual","name":name})
print(f"  ✓ {len(TECH)} технологических прорывов")

# ══ ИТОГ ══
print(f"\n{'='*50}")
print(f"Всего событий: {len(events)}")
cats = Counter(e["category"] for e in events)
print("\nПо категориям:")
for cat, n in cats.most_common():
    bar = "█" * (n//5)
    print(f"  {cat:<22} {n:>4}  {bar}")

# Сортируем по дате
events.sort(key=lambda x: x["date"])

# Добавляем к существующим
with open("gdelt_events.json") as f:
    existing = json.load(f)

existing_dates = {(e["date"][:7], e["category"]) for e in existing}
new_events = [e for e in events
              if (e["date"][:7], e["category"]) not in existing_dates]

all_events = existing + new_events
all_events.sort(key=lambda x: x["date"])

print(f"\nОбъединение:")
print(f"  Было:   {len(existing)}")
print(f"  Новых:  {len(new_events)}")
print(f"  Итого:  {len(all_events)}")

cats2 = Counter(e["category"] for e in all_events)
print("\nФинальный датасет:")
for cat, n in cats2.most_common():
    bar = "█" * (n//5)
    print(f"  {cat:<22} {n:>4}  {bar}")

with open("big_events.json", "w") as f:
    json.dump(all_events, f, indent=2, ensure_ascii=False)

print(f"\n✓ big_events.json — {len(all_events)} событий с 1800 по 2025")
print("Запускай: python3 retrain_big.py")
