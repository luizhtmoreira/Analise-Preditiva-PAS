from dataclasses import dataclass

@dataclass
class ExamStats:
    m_p1: float; dp_p1: float
    m_p2: float; dp_p2: float
    m_red: float; dp_red: float

# Gerado automaticamente via análise do PAS_MESTRE_LIMPO_FINAL.csv
# Chave: (Ano da Prova, Etapa)
OFFICIAL_STATS = {
    (2016, 1): ExamStats(m_p1=4.421, dp_p1=2.782, m_p2=24.246, dp_p2=13.169, m_red=6.074, dp_red=2.669),
    (2017, 1): ExamStats(m_p1=3.316, dp_p1=2.859, m_p2=27.408, dp_p2=13.417, m_red=6.222, dp_red=2.639),
    (2017, 2): ExamStats(m_p1=4.516, dp_p1=2.806, m_p2=20.403, dp_p2=11.959, m_red=6.100, dp_red=2.246),
    (2018, 1): ExamStats(m_p1=3.135, dp_p1=2.651, m_p2=25.938, dp_p2=14.166, m_red=5.919, dp_red=2.406),
    (2018, 2): ExamStats(m_p1=3.101, dp_p1=2.907, m_p2=24.410, dp_p2=12.196, m_red=7.053, dp_red=1.641),
    (2018, 3): ExamStats(m_p1=4.550, dp_p1=2.277, m_p2=28.433, dp_p2=14.280, m_red=6.848, dp_red=1.724),
    (2019, 1): ExamStats(m_p1=4.117, dp_p1=2.693, m_p2=27.041, dp_p2=13.935, m_red=6.657, dp_red=2.373),
    (2019, 2): ExamStats(m_p1=4.184, dp_p1=2.291, m_p2=25.439, dp_p2=12.696, m_red=6.844, dp_red=1.752),
    (2019, 3): ExamStats(m_p1=3.268, dp_p1=2.003, m_p2=24.678, dp_p2=11.531, m_red=7.013, dp_red=1.772),
    (2020, 1): ExamStats(m_p1=2.328, dp_p1=2.470, m_p2=24.784, dp_p2=13.366, m_red=5.743, dp_red=2.637),
    (2020, 2): ExamStats(m_p1=4.528, dp_p1=2.456, m_p2=29.006, dp_p2=12.915, m_red=7.032, dp_red=1.903),
    (2020, 3): ExamStats(m_p1=4.018, dp_p1=2.114, m_p2=28.199, dp_p2=12.847, m_red=6.972, dp_red=1.783),
    (2021, 1): ExamStats(m_p1=4.373, dp_p1=3.277, m_p2=21.806, dp_p2=12.448, m_red=5.984, dp_red=2.908),
    (2021, 2): ExamStats(m_p1=3.328, dp_p1=2.176, m_p2=25.349, dp_p2=11.911, m_red=7.125, dp_red=1.839),
    (2021, 3): ExamStats(m_p1=3.284, dp_p1=1.791, m_p2=23.678, dp_p2=12.372, m_red=7.009, dp_red=1.947),
    (2022, 1): ExamStats(m_p1=3.604, dp_p1=3.005, m_p2=20.709, dp_p2=13.581, m_red=5.888, dp_red=2.779),
    (2022, 2): ExamStats(m_p1=4.861, dp_p1=2.655, m_p2=22.192, dp_p2=11.832, m_red=7.505, dp_red=1.645),
    (2022, 3): ExamStats(m_p1=3.361, dp_p1=1.849, m_p2=26.385, dp_p2=13.146, m_red=7.482, dp_red=1.752),
    (2023, 2): ExamStats(m_p1=3.739, dp_p1=2.238, m_p2=30.348, dp_p2=13.252, m_red=6.937, dp_red=1.972),
    (2023, 3): ExamStats(m_p1=3.857, dp_p1=1.947, m_p2=27.258, dp_p2=12.923, m_red=6.893, dp_red=1.984),
    (2024, 3): ExamStats(m_p1=3.768, dp_p1=2.178, m_p2=32.086, dp_p2=14.128, m_red=7.579, dp_red=1.730),
}