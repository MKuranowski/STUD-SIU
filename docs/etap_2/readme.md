# SIU Projekt - Etap 2 - Zespół 3

* Konrad Wojda, 310990
* Mikołaj Kuranowski, 310779
* Bartłomiej Krawczyk, 310774
* Mateusz Brzozowski, 310608

## Uruchomienie środowiska

Zestawienie środowiska treningowego było najtrudniejszym punktem tego etapu. Każda osoba z zespołu miała problem z uruchomieniem lub treningiem żółwi. Decyzje architekturalne ROSa powodują wyciek pamięci w zależności systemu operacyjnego hosta. Dodatkowe
problemy z wydajnością środowiska zmusiły nas do przepisania kodu od podstaw.

Nasz kod źródłowy zawiera gruntowną przebudowę nie tylko kodu trenującego, ale również reimplementację środowiska - która zawiera autorski interfejs graficzny wykorzystujący pygame. Pozwala to na uruchamianie projektu w prostych w obsłudze środowiskach wirtualnych Pythona (virtualenv). Implementacja pozwala na wyłączenie środowiska graficznego. Aby spełnić wymagania projektu, kod zawiera implementację środowiska turtlesim. Zmienna środowiskowa `SIU_BACKEND` określa sposób uruchamiania - możliwe wartości to: `pygame`, `simple` (bez interfejsu graficznego), `ros`.

Ze względu na to, że ROS oraz turtlesim wymuszają wykorzystywanie starej wersji Pythona (3.8), na którą nie są dostępne najnowsze wydania Tensorflow i Keras, żadna osoba z zespołu nie była w stanie uruchomić uczenia z przyspieszeniem sprzętowym karty graficznej. Przy wykorzystaniu nowszych wersji tych bibliotek, takie przyspieszenie udało się uruchomić, jednakże modele wyuczone w ten sposób nie są kompatybilne z wymuszoną wersją Tensorflow.

Tak więc w przesłanym przez nas kodzie wykorzystujemy wymuszone starsze wersje Tensorflow oraz Keras, jednakże istnieje tutaj możliwość uruchomienia z nowszymi wersjami. 

Trenowanie modeli odbywa się zgodnie ze skryptem w pliku `turtle_estimator.py`. Istnieje możliwość uruchomienia wielu wątków aby uczyć wiele modeli jednocześnie. Przy zapisywaniu modeli do nazwy pliku dodajemy hash w celu łatwiejszej identyfikacji każdego z nich przy analizie wyników. Uruchomienie modeli odbywa się przez skrypt w pliku `play_single.py`. Aby uruchomić najlepszy model przygotowany został skrypt `run_best_single.sh`.

### Uruchomienie w przesłanym obrazie Docker

```shell
docker run --rm --name siu-24l-z3 -p 8080:80 -e RESOLUTION=1920x1080 -m 8g --ulimit nofile=1024:524288 bartlomiejkrawczyk/siu-20.04:ETAP_2
```

W terminalu w wirtualnym pulpicie (<http://localhost:8080>), w katalogu domowym (`/root`):

```shell
./run_best_single.sh
```

Uruchomienie modelu z wykorzystaniem autorskiej implementacji środowiska w pygame:
```shell
SIU_BACKEND=pygame python3 -m src.play_single models/dqns-7704f2-Gr7_Cr200_Sw4.0_Sv-20_Sf-15.0_Dr4_Oo-20.0_Ms20_Pb6_D0.85_M20000_m4000_B32_U20_T4.h5
```

## Najlepszy scenariusz

Najlepszy model znajduje się w pliku `models/dqns-7704f2-Gr7_Cr200_Sw4.0_Sv-20_Sf-15.0_Dr4_Oo-20.0_Ms20_Pb6_D0.85_M20000_m4000_B32_U20_T4.h5`. Bez problemu jest on w stanie wykonać kilkanaście kółek z rzędu za każdym uruchomieniam, co pozwala osiągnąć wartość parametru η >> 1. Prezentuje to drugi z poniższych obrazów. 

Graficzna reprezentacja najlepszego scenariusza:

![image](https://hackmd.io/_uploads/SksXufnZA.png)

![image](https://hackmd.io/_uploads/H1f2S7nWR.png)

Ustawione parametry w najlepszym modelu:
- rozdzielczość siatki: 7,
- rozdzielczość kamery: 200,
- współczynnik nagrody za jazdę zgodnie z sugerowanym kierunkiem: 4,
- współczynnik kary za jazdę przciwnie do sugerowanego kierunku: -20,
- współczynnik kary za zbyt szybką jazdę: -15,
- współczynnik nagrody za zbliżanie się do celu: 4,
- kara za wypadnięcie z trasy: -20,
- maksymalna liczba kroków agentu podczas uczenia: 20,
- dyskonto: 0.85.


## Eksperymenty

W ramach pierwszego etapu wytrenowaliśmy w sumie 65 modeli z losowo wybieranymi parametrami. Jako wskaźnik nauczenia modelu ("reward") wybraliśmy sumę ogólnej nagrody razy jeden plus liczba kółek wykonanych przez model do kolizji lub do osiągnięcia 4000 kroków. Poniżej znajduje się zestawienie tego wskaźnika w zależności od każdego parametru.

![image](https://hackmd.io/_uploads/BkNccQhWC.png)

Model o najwyższym wskaźniku nauczenia w praktyce wygląda na przeuczony i należy go wykluczyć z dalszej analizy. Dodatkowo posiłkując uśrednionymi wartościami wskaźnika możemy wskazać teoretycznie najlepszy zestaw parametrów:
- rozdzielczość siatki: 7,
- rozdzielczość kamery: 200,
- współczynnik nagrody za jazdę zgodnie z sugerowanym kierunkiem: 4,
- współczynnik kary za jazdę przciwnie do sugerowanego kierunku: -20,
- współczynnik kary za zbyt szybką jazdę: -15,
- współczynnik nagrody za zbliżanie się do celu: 4,
- kara za wypadnięcie z trasy: -15,
- maksymalna liczba kroków agentu podczas uczenia: 40,
- dyskonto: 0.85.

Jak widać, są to parametry zbliżone do parametrów modelu, który wybraliśmy jako najlepszy.
