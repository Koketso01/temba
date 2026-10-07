# Testing TEMBA

Thank you for trying TEMBA before its first public release. A test takes
about half an hour, most of it installation.

## 1. Install

```bash
git clone https://github.com/koketso01/temba.git
cd temba
./install.sh          # conda environment "temba" with everything, SoFiA-2 included
conda activate temba
```

`install.sh` needs conda (Miniforge is fine) and git. If you already have
SoFiA-2, use `./install.sh --no-sofia` and give its path in the survey file.

## 2. The built-in tests (a few seconds)

```bash
temba selftest
```

## 3. The demo survey (a few minutes)

```bash
temba demo temba_demo
cd temba_demo
temba check survey.yaml
temba run survey.yaml
temba status survey.yaml      # Demo: 2 realisations, about 80 injected
temba analyse survey.yaml     # results in work/analysis/
```

## 4. Optional: your own data

```bash
temba init my_survey.yaml     # fill in your fields
temba check my_survey.yaml
temba run my_survey.yaml
```

## 5. Tell us how it went

```bash
temba report                  # or: temba report my_survey.yaml
```

Then either open an issue on GitHub ("Test report" template) or email
`temba_report.txt` with a few lines on what worked and what did not to
Koketso Mophahlane (kmophahlane@gmail.com).
