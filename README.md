## Środowisko Deweloperskie

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

W repozytorium znajduje się reimplementacja środowiska nie wymagająca ROSa. Do uruchomienia
potrzebne wystarczy zwykłe wirtualne środowisko Pythona (venv), z zależnościami z pliku requirements.txt.

Wersja PyTorch z PyPI zawiera tylko wsparcie dla CUDA. Aby zapewnić działanie na GPU AMD,
należy odinstalować PyTorch z PyPI (`pip uninstall torch`) i zainstalować PyTorch z innego repozytorium:
`pip install torch --index-url https://download.pytorch.org/whl/nightly/rocm6.0`.

Aby PyTorch zadziałał na Radeonie 7800 XT dodatkowo trzeba ustawić zmienną środowiskową
`HSA_OVERRIDE_GFX_VERSION=11.0.0`.
