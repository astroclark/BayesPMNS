# $Id: Makefile,v 1.4 2012/01/20 20:10:43 jclark Exp $
default: bns_bursts_clark


bns_bursts_clark:
	rm -f bns_bursts_clark.aux bns_bursts_clark.bbl
	pdflatex bns_bursts_clark
	bibtex bns_bursts_clark
	pdflatex bns_bursts_clark
	pdflatex bns_bursts_clark

clean:
	@echo "Cleaning directory of backups and logs"
	rm -f *~ *.log *.aux *.dvi *.lot *.lof *.toc *.bbl *.blg *.out *.end *Notes*bib *.nav *.snm *.pdf
