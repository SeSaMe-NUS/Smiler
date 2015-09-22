#include "csvstream.h"

/**
 * Default constructor
 */
csvstream::csvstream()
{
}

/**
 * Destructor
 */
csvstream::~csvstream()
{
}

/**
 * Read in a matrix from a CSV file
 * 
 * @param M         The returned matrix with P rows and N columns, where P is the number 
 *                  of lines of the CSV file and N is the number of values (separated
 *                  by commas) on each line   
 * @param filename  The name of a file containing the elements of a P by N matrix as 
 *                  comma separated values (N values per lines, P lines)
 * @return          1 if there was an error, 0 otherwise
 */
int csvstream::read(mat &M, const string filename) 
{
	vector < vector <string> > data;
	vector<string> row;
	string element, delimiters = ",\n\r";
	
	char ch;
	
	// Open file for reading
	debug_msg("Opening CSV file");
	ifstream fin(filename.c_str(), ios::in);
	
	// Make sure file is open
	debug_msg("Make sure file is open");
	if (!fin.is_open()) {
		cerr << "Error opening file '" << filename << "'" << endl;
		return 1;
	}

	debug_msg("Read data in");
	while( fin.read( (char*)&ch, 1 ) )
	{
		// If character read is not a delimiter, add to element string
		if( delimiters.find_first_of(ch) == delimiters.npos ) {
			element += ch;
		}
		else {
			// If character is a comma or end-of-line, add current element to 
			// data and reset element
			if( ch != '\r' ) {  
				row.push_back( element );
				element = "";

				// If character is end-of-line, add a new row to data
				if( ch == '\n' ) {
					data.push_back( row );
					row = vector<string>();
				}
			}
		}
	}

	// If the last element is not empty (happens when the last line
	// of the file does not end with end-of-line), add the last element
	// to data
	debug_msg("If necessary, add final element.");
	if( element.size() > 0 ) {
		row.push_back( element );
		data.push_back( row );
	}

	// Close file
	debug_msg("Close file.");
	fin.close();

	// Allocate matrix
	if (data.size() == 0 || data[0].size() == 0) {         // Empty matrix: return
		debug_msg("Empty matrix"); 
		M.set_size(0,0);
		return 1;
	}
	else {
		debug_msg("Allocate matrix (" << data.size() << "," << data[0].size() << ")");
		  //! Set size of matrix. If copy = true then keep the data before resizing.
		//M.set_size((int) data.size(), (int) data[0].size(), false);//comment by jingbo
		M.set_size((int) data.size(), (int) data[0].size());
	}
	
	// Convert 2D array of strings to matrix of double
	// RB: if atof fails, it returns 0.0 - how do we check everything went fine???
	// RB: There must be an error flag set somewhere...
	debug_msg("Convert data to matrix.");
	for( unsigned int i = 0; i < data.size(); i++ ) {
		for( unsigned int j = 0; j < data[i].size(); j++ ){
			M(i,j) = atof( data[i][j].c_str() );
		}
	}
	
	return 0;
}


/**
 * Writes a matrix to a CSV file
 * 
 * @param M         A matrix with P rows and N columns
 * @param filename  The name of a file where the elements M are written as  
 *                  comma separated values (N values per lines, P lines)
 * @return          dan1 if there was an error, 0 otherwise
 */
int csvstream::write(const mat M, const string filename, int decimals) 
{
	int i,j;
	
	// assert(M.rows() > 0 && M.cols() > 0);
  
	// Open file for writing
	debug_msg("Open file for writing.");
	ofstream fout (filename.c_str(), ios::out);

	// Make sure file is open
	debug_msg("Make sure file is open.");
	if (!fout.is_open()) {
		cerr << "Error opening file " << filename << " for writing." << endl;
		return 1;
	}

	// Set precision
	fout.precision(decimals);
	fout.setf(ios::fixed,ios::floatfield);
	
	// Write data
	debug_msg("Write data to file.");
	//for ( i=0; i<M.rows(); i++ ) {//comment by jingbo
	//	for ( j=0; j<M.cols()-1; j++ ) {
	for ( i=0; i<M.n_rows; i++ ) {//add by jingbo
			for ( j=0; j<M.n_cols-1; j++ ) {
			fout << M(i,j) << ", ";
		}
		//fout << M(i, M.cols()-1) << "\r\n";//comment by jingbo
		fout << M(i, M.n_cols-1) << "\r\n";
	}

	// Close file
	fout.close();

	return 0;
}
