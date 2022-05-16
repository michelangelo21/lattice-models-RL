# Ideas
    - AlphaZero GPU

# Problems
    - action-space for XYmodel
        Try tuple of discrete action space for choosing particle, and
            continuous/discretized for angle of rotation.

Uczy się układać Isinga tylko na jedną stronę (zawsze tylko +1 lub -1).

# Differentiable
- liczenie gradientu nie tylko po jednym kroku, ale po 2, 3, 4 itp.
(dE_dx) = dE_d(x+xbar) * d(x+xbar)_dx, nwm czy ma sens, bo plansza nie ma "pamięci", nie jest zależna od czasu, interesuje ją tylko obecny stan i nie pamięta skąd do niego doszła


# TODO
- [x] Ising1D
- [x] CNN
    - [] zobaczyć, czy da się sieć CNN nauczoną na mniejszym układzie przenieść na większy
- [] ResNet