" ATP project vim script: Sun Aug 30, 2015 at 11:11 AM +0200.

let b:atp_MainFile = 'main.tex'
let g:atp_mapNn = 0
let b:atp_autex = 1
let b:atp_TexCompiler = 'pdflatex'
let b:atp_TexOptions = '-synctex=1'
let b:atp_TexFlavor = 'tex'
let b:atp_auruns = '1'
let b:atp_ReloadOnError = '1'
let b:atp_OutDir = '/home/pascal/ownCloud/MA/Code/tex'
let b:atp_OpenViewer = '1'
let b:atp_XpdfServer = 'main'
let b:atp_Viewer = 'evince'
let b:TreeOfFiles = {'./ch_ladder_operators.tex': [{}, 76], './title.tex': [{}, 54], './ch_splitting_morse.tex': [{}, 68], './ch_introduction.tex': [{}, 64], './ap_integrals.tex': [{}, 85], './ch_parameters.tex': [{}, 80], './ch_analytic_solution.tex': [{}, 72]}
let b:ListOfFiles = ['./header_math.tex', './header_gfx.tex', './header_algs.tex', './header_listing.tex', './title.tex', './ch_introduction.tex', './ch_splitting_morse.tex', './ch_analytic_solution.tex', './ch_ladder_operators.tex', './ch_parameters.tex', './ap_integrals.tex', 'tt.bib']
let b:TypeDict = {'./ch_analytic_solution.tex': 'input', './ch_ladder_operators.tex': 'input', './header_listing.tex': 'preambule', './ch_splitting_morse.tex': 'input', './header_math.tex': 'preambule', './header_algs.tex': 'preambule', './ch_introduction.tex': 'input', './header_gfx.tex': 'preambule', './ap_integrals.tex': 'input', './title.tex': 'input', './ch_parameters.tex': 'input', 'tt.bib': 'bib'}
let b:LevelDict = {'./ch_analytic_solution.tex': 1, './ch_ladder_operators.tex': 1, './header_listing.tex': 1, './ch_splitting_morse.tex': 1, './header_math.tex': 1, './header_algs.tex': 1, './ch_introduction.tex': 1, './header_gfx.tex': 1, './ap_integrals.tex': 1, './title.tex': 1, './ch_parameters.tex': 1, 'tt.bib': 1}
let b:atp_BibCompiler = 'bibtex'
let b:atp_StarEnvDefault = ''
let b:atp_StarMathEnvDefault = ''
let b:atp_updatetime_insert = 4000
let b:atp_updatetime_normal = 2000
let b:atp_LocalCommands = ['\blue', '\clearemptydoublepage', '\assign', '\rassign', '\seteq', '\of{', '\ofs{', '\norm{', '\tmop{', '\id', '\kron{', '\conj{', '\inv', '\T', '\herm', '\tr', '\ft{', '\ift{', '\fft{', '\ifft{', '\dotp{', '\bigO{', '\laplace', '\di{', '\diff', '\pdiff{', '\python{', '\cpp{', '\cppinline', '\gnuR{']
let b:atp_LocalEnvironments = []
let b:atp_LocalColors = ['linkcol', 'citecol', 'gray']
