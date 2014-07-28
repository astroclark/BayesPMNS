# $Id: Makefile,v 1.4 2012/01/20 20:10:43 jclark Exp $
default: bayes_pmns


bayes_pmns:
	rm -f bayes_pmns.aux bayes_pmns.bbl
	latex bayes_pmns
	bibtex bayes_pmns
	latex bayes_pmns
	latex bayes_pmns
	dvipdf bayes_pmns

clean:
	@echo "Cleaning directory of backups and logs"
	rm -f *~ *.log *.aux *.dvi *.lot *.lof *.toc *.bbl *.blg *.out *.end *Notes*bib
