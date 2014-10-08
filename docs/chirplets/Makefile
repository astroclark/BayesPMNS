# $Id: Makefile,v 1.4 2012/01/20 20:10:43 jclark Exp $
default: chirplets


chirplets:
	rm -f chirplets.aux chirplets.bbl
	latex chirplets
	bibtex chirplets
	latex chirplets
	latex chirplets
	dvipdf chirplets

clean:
	@echo "Cleaning directory of backups and logs"
	rm -f *~ *.log *.aux *.dvi *.lot *.lof *.toc *.bbl *.blg *.out *.end *Notes*bib
