import json, time
from collections import Counter

print("Расширенный исторический датасет — 200+ событий\n")

HISTORICAL = [
    ("1979-01-01","geopolitical"),("1979-02-11","geopolitical"),
    ("1979-12-24","geopolitical"),("1980-09-22","geopolitical"),
    ("1981-10-06","geopolitical"),("1982-06-06","geopolitical"),
    ("1983-10-23","geopolitical"),("1984-10-31","geopolitical"),
    ("1985-03-11","geopolitical"),("1986-04-15","geopolitical"),
    ("1988-07-03","geopolitical"),("1989-06-04","geopolitical"),
    ("1989-11-09","geopolitical"),("1990-08-02","geopolitical"),
    ("1991-01-17","geopolitical"),("1991-08-19","geopolitical"),
    ("1991-12-25","geopolitical"),("1992-04-05","geopolitical"),
    ("1993-02-26","geopolitical"),("1994-04-06","geopolitical"),
    ("1995-11-04","geopolitical"),("1996-09-27","geopolitical"),
    ("1997-07-01","geopolitical"),("1998-08-07","geopolitical"),
    ("1999-03-24","geopolitical"),("1999-08-09","geopolitical"),
    ("2001-09-11","geopolitical"),("2003-03-20","geopolitical"),
    ("2004-03-11","geopolitical"),("2005-07-07","geopolitical"),
    ("2006-10-09","geopolitical"),("2007-12-27","geopolitical"),
    ("2008-08-08","geopolitical"),("2011-02-17","geopolitical"),
    ("2011-05-02","geopolitical"),("2012-07-18","geopolitical"),
    ("2013-04-15","geopolitical"),("2013-08-21","geopolitical"),
    ("2014-02-22","geopolitical"),("2014-03-18","geopolitical"),
    ("2014-07-17","geopolitical"),("2015-01-07","geopolitical"),
    ("2015-11-13","geopolitical"),("2016-07-15","geopolitical"),
    ("2017-04-07","geopolitical"),("2018-04-14","geopolitical"),
    ("2019-01-03","geopolitical"),("2020-01-03","geopolitical"),
    ("2021-01-06","geopolitical"),("2021-08-15","geopolitical"),
    ("2022-02-24","geopolitical"),("2023-10-07","geopolitical"),
    ("2024-04-01","geopolitical"),("2024-10-07","geopolitical"),
    ("1979-10-06","economic_crisis"),("1981-06-01","economic_crisis"),
    ("1982-08-12","economic_crisis"),("1987-10-19","economic_crisis"),
    ("1990-08-01","economic_crisis"),("1992-09-16","economic_crisis"),
    ("1994-12-20","economic_crisis"),("1997-07-02","economic_crisis"),
    ("1997-10-27","economic_crisis"),("1998-08-17","economic_crisis"),
    ("1998-09-23","economic_crisis"),("2000-03-10","economic_crisis"),
    ("2001-12-02","economic_crisis"),("2002-07-19","economic_crisis"),
    ("2007-02-27","economic_crisis"),("2007-08-09","economic_crisis"),
    ("2008-03-14","economic_crisis"),("2008-09-15","economic_crisis"),
    ("2008-10-08","economic_crisis"),("2009-03-09","economic_crisis"),
    ("2010-05-10","economic_crisis"),("2011-08-05","economic_crisis"),
    ("2012-06-01","economic_crisis"),("2014-10-15","economic_crisis"),
    ("2015-08-24","economic_crisis"),("2016-01-20","economic_crisis"),
    ("2018-12-24","economic_crisis"),("2020-02-24","economic_crisis"),
    ("2020-03-20","economic_crisis"),("2022-01-24","economic_crisis"),
    ("2023-03-10","economic_crisis"),("2023-09-20","economic_crisis"),
    ("1980-05-18","natural_disaster"),("1982-03-29","natural_disaster"),
    ("1985-09-19","natural_disaster"),("1986-04-26","natural_disaster"),
    ("1988-12-07","natural_disaster"),("1989-10-17","natural_disaster"),
    ("1991-06-15","natural_disaster"),("1994-01-17","natural_disaster"),
    ("1995-01-17","natural_disaster"),("1999-08-17","natural_disaster"),
    ("1999-09-21","natural_disaster"),("2003-08-01","natural_disaster"),
    ("2004-12-26","natural_disaster"),("2005-08-29","natural_disaster"),
    ("2005-10-08","natural_disaster"),("2008-05-02","natural_disaster"),
    ("2008-05-12","natural_disaster"),("2010-01-12","natural_disaster"),
    ("2010-04-20","natural_disaster"),("2011-02-22","natural_disaster"),
    ("2011-03-11","natural_disaster"),("2012-10-29","natural_disaster"),
    ("2013-11-08","natural_disaster"),("2015-04-25","natural_disaster"),
    ("2016-08-24","natural_disaster"),("2017-08-25","natural_disaster"),
    ("2017-09-07","natural_disaster"),("2018-09-06","natural_disaster"),
    ("2018-11-08","natural_disaster"),("2021-07-14","natural_disaster"),
    ("2023-02-06","natural_disaster"),("2024-01-01","natural_disaster"),
    ("2024-09-27","natural_disaster"),
    ("1981-06-05","epidemic"),("1986-01-01","epidemic"),
    ("1993-01-01","epidemic"),("2002-11-01","epidemic"),
    ("2003-03-15","epidemic"),("2009-04-09","epidemic"),
    ("2012-09-01","epidemic"),("2014-03-23","epidemic"),
    ("2015-05-20","epidemic"),("2016-02-01","epidemic"),
    ("2018-08-01","epidemic"),("2019-12-31","epidemic"),
    ("2020-01-30","epidemic"),("2020-03-11","epidemic"),
    ("2020-11-09","epidemic"),("2021-05-05","epidemic"),
    ("2021-11-26","epidemic"),("2022-05-23","epidemic"),
    ("2024-06-01","epidemic"),
    ("1981-08-12","tech_breakthrough"),("1983-01-01","tech_breakthrough"),
    ("1985-10-18","tech_breakthrough"),("1989-03-12","tech_breakthrough"),
    ("1991-08-06","tech_breakthrough"),("1993-04-22","tech_breakthrough"),
    ("1995-08-09","tech_breakthrough"),("1996-02-10","tech_breakthrough"),
    ("1997-05-11","tech_breakthrough"),("1998-09-04","tech_breakthrough"),
    ("2000-01-14","tech_breakthrough"),("2001-10-23","tech_breakthrough"),
    ("2004-02-04","tech_breakthrough"),("2007-01-09","tech_breakthrough"),
    ("2008-09-10","tech_breakthrough"),("2009-01-03","tech_breakthrough"),
    ("2010-06-24","tech_breakthrough"),("2011-10-05","tech_breakthrough"),
    ("2012-08-25","tech_breakthrough"),("2013-12-19","tech_breakthrough"),
    ("2016-03-09","tech_breakthrough"),("2016-11-30","tech_breakthrough"),
    ("2017-12-17","tech_breakthrough"),("2020-11-09","tech_breakthrough"),
    ("2022-11-30","tech_breakthrough"),("2023-03-14","tech_breakthrough"),
    ("2023-07-18","tech_breakthrough"),("2024-02-15","tech_breakthrough"),
    ("2024-05-13","tech_breakthrough"),("2024-09-12","tech_breakthrough"),
]

events = [{"date": d, "category": c, "goldstein": -7.0, "tone": -3.0}
          for d, c in HISTORICAL]

print(f"Событий: {len(events)}")
cats = Counter(e["category"] for e in events)
print("\nПо категориям:")
for cat, n in cats.most_common():
    bar = "█" * n
    print(f"  {cat:<22} {n:>3}  {bar}")

with open("gdelt_events.json", "w") as f:
    json.dump(events, f, indent=2, ensure_ascii=False)

print("\n✓ gdelt_events.json")
print("Запускай: python3 retrain_full.py")
