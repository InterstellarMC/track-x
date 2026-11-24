# Exclusive pixel and outer hit ownership

The joint arm relaxes the original pixel ownership reservation. Seeds compete by the best cost among all offered alternatives, and can lose their original hits. Greedy assignment keeps maximum one candidate per original and one owner per hit. By comparison, global in the benchmark uses the frozen Phase IV outer-path prefix packer and original pixel seeds. Neither procedure solves exact weighted set packing.

| method | score | mixed | severe | fake | retention |
|---|---|---|---|---|---|
| geometry | 0.220071 | 0.430569 | 0.003700 | 0.511905 | 1.000000 |
| local | 0.223755 | 0.393249 | 0.002208 | 0.503056 | 1.000000 |
| global | 0.223589 | 0.382836 | 0.001492 | 0.503603 | 1.000000 |
| tight | 0.222055 | 0.378905 | 0.001714 | 0.506083 | 1.000000 |
| strict | 0.220206 | 0.291632 | 0.002082 | 0.429706 | 0.997936 |
| replace | 0.224635 | 0.386733 | 0.002306 | 0.501171 | 0.895742 |
| beam | 0.224720 | 0.386599 | 0.002325 | 0.500755 | 0.897439 |
| joint | 0.225026 | 0.363818 | 0.001855 | 0.484687 | 0.792088 |
