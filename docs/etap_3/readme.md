# SIU Projekt - Etap 3 - Zespół 3

* Konrad Wojda, 310990
* Mikołaj Kuranowski, 310779
* Bartłomiej Krawczyk, 310774
* Mateusz Brzozowski, 310608

## Plansza z celami ruchu oraz polami startowymi

Zmieniliśmy tor w taki sposób, aby występowało na nim skrzyżowanie oraz dostosowaliśmy scenariusze tak, aby występowała realna szansa kolizji na skrzyżowaniu.

![Group 1 (1)](https://hackmd.io/_uploads/ryDBcEUE0.png)

### Plik ze scenariuszami

| id | agents_no | start_left | start_right | start_bottom | start_top | goal_x | goal_y |
|----|-----------|------------|-------------|--------------|-----------|--------|--------|
|1|2|54.09|57.5|10.18|14.73|8.64|12.05|
|1|2|7.09|10.5|12.18|16.73|27.14|40.09|
|1|2|27.5|32.05|36.68|40.09|62.77|19.59|
|1|2|63.36|67.91|15.95|19.36|82|18.81|
|1|2|71.59|76.13|40.05|43.45|55.59|36|
|1|2|54.09|57.5|31|35.54|55.82|14.73|


## Dostosowanie do uczenia wieloagentowego

Jak wspomnieliśmy w sprawozdaniu z poprzedniego etapu, w związku z problemami z uruchomieniem i zrozumieniem udostępnionego kodu, zaimplementowaliśmy wszystko od podstaw. Dzięki temu nie musieliśmy wprowadzać wielu zmian w kodzie. Nie potrzebowaliśmy osobnej klasy `EnvMulti`, nasza reprezentacja środowiska była zaimplementowana poprawnie zarówno dla sytuacji jedno-, jak i wieloagentowych.

Zaimplemntowaliśmy klasę `DQNMulti`, aby działała tak jak ta udostępniona, jednak w znacznie
czytelniejszy sposób. Dzięki poprawnej dekompozycji problemu w naszej implementacji, `DQNMulti` nadpisuje jedynie tworzenie modelu i tensora wejściowego do sieci (aby możliwe było przekazanie macierzy zajętości
kamery żółwia) oraz główna pętla uczenia, aby możliwe było równoległe uczenie wielu epizodów na raz.

Największej zmiany wymagała jednak klasa `PlayMulti`, konkretnie sposób ewaluacji modelu. Wyznacznik nauczenia modelu to teraz Σ a ∈ agents: total_rewardₐ * (1 + total_sectionsₐ). Liczba wykonaych okrążeń zależy od sekcji w której agent został umieszczony na samym początku. Ewaluacja przerywana jest gdy połowa żółwi wyjedzie poza trasę lub zderzy się z innym żółwiem. Niestety napotkaliśmy duże problemy z powtarzalnością wyników, więc jako ostateczny wyznacznik bierzemy średnią z trzech uruchomień.

## Uczenie

Uczenie rozpoczęliśmy od następujących pól startowych:

![image](https://hackmd.io/_uploads/SJBCIHUVC.png)

Następnie kolejno próbowaliśmy różnych podejść do nauczania, korygując je zgodnie z napotkanymi problemami:

1. **dqnm** - uczenie analogiczne do etapu pierwszego, jedynie z większą liczbą agentów. Kolizje zostały włączone od razu.

Żółwie w modelach nauczonych podejściem `dqnm` w większości nie zdołały nauczyć się poprawnie poruszać po planszy. W modelach, które to osiągnęły, żółwie za wszelką cenę unikały skrzyżowania, co prowadziło do częstego wyjeżdżania poza tor, bądź skręcania w złą stronę na skrzyżowaniu (co powodowało brak możliwości osiągnięcia celu). 

Aby spróbować zapewnić, że żółwie w pierwszej kolejności nauczą się poruszać po planszy, a następnie dopiero unikać kolizji zastosowaliśmy kolejne podejście:

2. **dqnp** - rozpoczęcie nauki z wyłączonymi kolizjami, włączenie kolizji po 2000 epizodzie.

Żółwie nauczone podejściem **dqnp** nadal za wszelką cenę starały się unikać skrzyżowania (poruszały się po jego obrzeżach - tak jak przedstawiono na poniższym obrazku). Dodatkowo część z nich chciała jechać prosto do celu nie zważając na granicę toru. Dotyczyło to przede wszystkim żółwi jadących z pól statrowych `7` do celu przy polach startowych `1`.

![image](https://hackmd.io/_uploads/Bkto7LINA.png)

Aby rozwiązać ten problem zmieniliśmy punkty startowe i cele, tak aby znajdowały się one bezpośrednio przed i po skrzyżowaniu. Dodatkowo zdecydowaliśmy się na opóźnienie włączenia kolizji do epizodu numer 3000.


3. **dqnd** - w tym podejściu zmieniliśmy punkty startowe i zmniejszyliśmy liczbę scenariuszy (a co za tym idzie - liczbę żółwi). Włączenie kolizji nastąpiło po 3000 epizodzie. 

Nowa plansza:

![Group 1 (1)](https://hackmd.io/_uploads/ryDBcEUE0.png)


W tym modelu żółwie poruszały się znacznie lepiej po planszy, ale nadal żółwie jadące z pól startowych numer `6` jechały bezpośredniu do celu. Było to spowodowane błędem w jego definicji, co zostało poprawione w kolejnym podejściu. Dodatkowo zweryfikowaliśmy, jakie parametry były najlepsze i zdecydowaliśmy się na ich poprawienie. Kluczową rzeczą było zmniejszenie nagrody za jazdę w przód, ponieważ żółwie często skręcały na skrzyżowaniu zamiast jechać prosto. Wróciliśmy również do włączania kolizji po 2000 epizodów, ze względu na zbyt słabe unikanie kolizji.

4. **dqnc** - poprawione parametry, włączanie kolizji po 2000 epizodów.

Żółwie nauczone podejściem `dqnc` jeździły zdecydowanie najlepiej ze wszystkich opisanych wyżej podejść, jednak w dalszym ciągu zdarzało się im wyjechać poza tor oraz nie uniknąć kolizji z innymi żółwiami. 

## Wyniki

![Screenshot from 2024-05-30 20-48-04](https://hackmd.io/_uploads/HyYe5BI4R.png)

### Najlepszy wytrenowany model

Najlepszy wytrenowany model osiągnął wynik 97440,51026. 

Ustawione parametry w najlepszym modelu:

- rozdzielczość siatki: 7,
- rozdzielczość kamery: 200,
- współczynnik nagrody za jazdę zgodnie z sugerowanym kierunkiem: 0.1,
- współczynnik kary za jazdę przeciwnie do sugerowanego kierunku: -10,
- współczynnik kary za zbyt szybką jazdę: -10,
- współczynnik nagrody za zbliżanie się do celu: 16,
- kara za wypadnięcie z trasy: -20,
- maksymalna liczba kroków agenta podczas uczenia: 40,
- dyskonto: 0.80.

#### Uruchomienie modelu w Dockerze

```shell
docker run --rm --name siu-24l-z3 -p 8080:80 -e RESOLUTION=1920x1080 -m 8g --ulimit nofile=1024:524288 bartlomiejkrawczyk/siu-20.04:ETAP_3
```

W terminalu w wirtualnym pulpicie (<http://localhost:8080>), w katalogu domowym (`/root`):

```shell
./run_best_multi.sh
```

Uruchomienie modelu z wykorzystaniem autorskiej implementacji środowiska w pygame:
```shell
SIU_BACKEND=pygame python3 -m src.play_multi models/dqnc-ddbad7-Gr7_Cr200_Sw0.1_Sv-10.0_Sf-10.0_Dr16.0_Oo-20.0_Ms40_Pb6_D0.8_M20000_m4000_B32_U20_T4.h5
```

#### Wynik uruchomienia

![siu](https://hackmd.io/_uploads/rJlYaB8VC.png)


### Tabela wyznaczników (`dqn_multi_models.csv`)

Poniższa tabela przedstawia kilka najlepszych modeli, jakie udało się uzyskać z każdego sposobu uczenia.

wyznacznik nauczenia|sposób uczenia|hash|rozdzielczość siatki|rozdzielczość kamery|nagorda za jazdę w przód|kara za jazdę pod prąd|kara za przekraczanie prędkości|nagroda za zbliżanie się do celu|kara za wypadnięcie/kolizję|liczba kroków|dyskonto
-|-|-|-|-|-|-|-|-|-|-|-|
97440,51026|dqnc|ddbad7|7|200|0,1|-10|-10|16|-20|40|0,8
16000,6281|dqnc|3fce41|9|300|1|-15|-10|24|-20|60|0,9
12140,98347|dqnd|8db3d7|9|200|0,5|-15|-10|16|-30|40|0,8
9462,102114|dqnc|909984|7|200|0,1|-20|-10|32|-20|40|0,85
7980,100982|dqnc|4e3142|9|300|0,1|-10|-10|16|-20|60|0,85
2709,820363|dqnp|1627ab|9|250|2|-15|-15|8|-20|80|0,85
1946,007347|dqnm|5e83ba|7|250|2|-15|-15|8|-10|80|0,8

### Tabla wyznaczników przy podwojonej ilości żółwi

wyznacznik nauczenia|sposób uczenia|hash|rozdzielczość siatki|rozdzielczość kamery|nagorda za jazdę w przód|kara za jazdę pod prąd|kara za przekraczanie prędkości|nagroda za zbliżanie się do celu|kara za wypadnięcie/kolizję|liczba kroków|dyskonto
-|-|-|-|-|-|-|-|-|-|-|-
23342,7953|dqnc|ddbad7|7|200|0,1|-10|-10|16|-20|40|0,8
16009,96178|dqnd|8db3d7|9|200|0,5|-15|-10|16|-30|40|0,8
13530,01662|dqnc|3fce41|9|300|1|-15|-10|24|-20|60|0,9
12626,18775|dqnc|909984|7|200|0,1|-20|-10|32|-20|40|0,85
9492.669765|dqnc|4e3142|9|300|0.1|-10|-10|16|-20|60|0.85
7755,350917|dqnp|1627ab|9|250|2|-15|-15|8|-20|80|0,85
3407.096464|dqnm|5e83ba|7|250|2.0|-15|-15|8|-10|80|0.8

## Omówienie wyników

Subiektywny opis wyników uczenia przedstawiliśmy we wcześniejszej sekcji; reasumując, po wielu próbach udało nam się nauczyć model jeżdzący w miarę poprawnie po planszy, który jednak nadal ma problemy z unikaniem kolizji.

![Screenshot from 2024-05-30 20-47-10](https://hackmd.io/_uploads/r1Pe9H8EC.png)


### Analiza wpływu parametrów na jakość modeli

W poniższej analizie danych w głównej mierze posługiwaliśmy się średnią arytmetyczną. Najlepszy model osiągnął wyznacznik nauczenia ponad 90 tys., a drugi najelpszy nieco ponad 15 tys., zatem wartości średnich będą mocno zachwiane przez odstający, najlepszy model.

Dodatkowo, w związku z niepowodzeniem uczenia sposobami **dqnm** i **dqnp** raczej nie należy się przywiązywać do wynikających z nich korelacji.

#### Rozdzielczość siatki

![Screenshot from 2024-05-30 20-43-10](https://hackmd.io/_uploads/HkXZ5SLV0.png)

Istnieje pewna dodatnia korelacja między rozdzielczością siatki a jakością modeli, jednak nie jest to przeważający parametr. Większa rozdzielczość siatki negatywnie wpływa na czas trenowania i uruchomienia modeli.

#### Rozdzielczość kamery

![Screenshot from 2024-05-30 20-43-23](https://hackmd.io/_uploads/ByQbqrLVA.png)

W przypadku pierwszych sposobów nauczania raczej nie ma preferowanych rozdzielczości kamer,
jednak w przypadku sposobów **dqnc** i **dqnd** preferowana jest mniejsza rozdzielczość kamery.
Może to być zaskakujące na pierwszy rzut oka, jednak mniejsza rozdzielczość kamery przy zachowaniu tej samej rozdzielczości siatki powoduje wzrost rozdzielczości pojedynczej komórki kamery.

#### Nagroda za jazdę w przód

![Screenshot from 2024-05-30 20-44-04](https://hackmd.io/_uploads/SkNbcH8E0.png)

Niska nagroda za jazdę zgodnie z kieruniem sugerowanym przez planszę okazała się niezbędna do pozbycia niechcianego zachowania unikającego skrzyżowania: kolor skrzyżowania zawiera mnożnik 0 tej nagrody, podczas gdy reszta toru zawiera mnożnik 1, co powoduje unikanie skrzyżowania przez żółwie. Niski ogólny współczynnik nagrody za jazdę w przód koryguje to zachowanie.

#### Kara za jazdę pod prąd

![Screenshot from 2024-05-30 20-44-23](https://hackmd.io/_uploads/r1N-crIVA.png)

Nie uważamy wartości tego parametru za znaczący, żółwie szybko oduczają się jazdy pod prąd.

#### Kara za przekroczenie prędkości

![Screenshot from 2024-05-30 20-44-57](https://hackmd.io/_uploads/BJ4-5B84A.png)

Podobnie jak w przypadku kary za jazdę pod prąd, nie uważamy aby zmiana wartości tego parametru była znacząca. Maksymalna prędkość żółwia nie może przekroczyć sugerowanej prędkości zakodowanej w kolorze pikseli planszy ze względu na dopuszczone sterowania.

#### Nagroda za zbliżanie się do celu

![Screenshot from 2024-05-30 20-45-17](https://hackmd.io/_uploads/BkVb9HUVR.png)

Wartość tego parametru jest kluczowa do zblisanowania zmniejszonej nagrody za jazdę zgodnie z sugerowanym kierunkiem. Analiza danych sugeruje, że wartości około 16 dają najlepsze rezultaty.
Zbyt duża wartość współczynnika „zachęca” żółwie do jazdy w prostej linii do celu, mimo że powoduje to zwykle wyjechanie z toru.

#### Kara za wypadnięcie/kolizję

![Screenshot from 2024-05-30 20-45-40](https://hackmd.io/_uploads/Sy4-cHLNC.png)

Również nie uważamy, żeby wartość tego parametru była aż tak znacząca, ale jednak można zaobserwować lepsze zachowanie modeli przy większej karze za zderzenie/wypadnięcie z toru.

#### Liczba kroków

![Screenshot from 2024-05-30 20-45-56](https://hackmd.io/_uploads/BJeVZqrUNC.png)

20 kroków to zdecydowanie zbyt mała liczba aby żółwie mogły się czegokolwiek nauczyć. Z drugiej strony, 80 kroków powodowało znaczne wydłużenie procesu uczenia, zatem w ostatniej fazie uczenia ograniczyliśmy wartość tego parametry do 40 lub 60, bez dużej różnicy w zachowaniu modeli.

#### Dyskonto

![Screenshot from 2024-05-30 20-46-09](https://hackmd.io/_uploads/HJNZcrUNA.png)

Wykresy mogą sugerować negatywną parametru, tj. im mniejsze dyskonto tym lepszy model, jednak dalsza analiza danych tabelarycznych raczej skłania nas ku stwierdzeniu, że wartość tego parametru nie wpływa znacznie na sposób jazdy żółwi.
