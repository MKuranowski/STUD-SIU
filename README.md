## Środowisko Deweloperskie

### Dec Container

Ponieważ praca w wirtualnym pulpicie dostarczonym przez prowadzących
nie jest zbytnio wygodna, ten projekt wykorzystuje system [devcontainer](https://containers.dev/).
Otwarcie projektu z rozszerzeniem [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
do VS Code powinno zasugerować uruchomienie projektu w kontenerze (jeżeli nie - wybierz `Dev Containers: Reopen in Container`).
Budowanie konetera może potrwać kilka minut.

Uwaga! Aby przekierowanie okien z działało należy uruchomić polecenie `xhost +local:docker` na hoście
z jego każdym ponownym uruchomieniem.

Aby uruchomić symulację wykonaj `./do_run.sh siu_example.py`. Skrypt ten automatycznie uruchomi ROS
i kod w Pythonie. Argumenty skryptu są przekazywane verbatim do polecenia `python3`.

### Lokalnie

W repozytorium znajduje się reimplementacja środowiska żółwi, dzięki czemu instalacja ROSa
i turtlesima nie jest potrzebna. Wystarczy stworzyć venv i zainstalować potrzebne zależności:

```terminal
$ python -m venv .venv
$ . .venv/bin/activate
$ pip install -Ur requirements.cpu_only.txt
```

Jeżeli posiadasz kartę graficzną nvidii, zmień plik z wymaganiami w ostatnim poleceniu na
`requirements.cuda.txt`.

Dostępne są 3 środowiska źółwi: `ros` (domyślny, jeżeli dostępny), `simple` (domyślny,
jeżeli nie ma ROSa) oraz `pygame`. Wybór można nadpisać za pomocą zmiennej środowiskowej
`SIU_BACKEND`. Uwaga! Środowisko `simple` nie ma interfejsu graficznego.
