#pragma once
#include <string>
#include <algorithm>
#include <Windows.h>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

// process the path to get the right format 
std::string getDirectoryPath(std::string path) {
	std::replace(path.begin(), path.end(), '\\', '/');
	int lastSlashIndex = path.find_last_of('/', (int)path.size());
	if (lastSlashIndex < (int)path.size() - 1)
		path += "/";
	return path;
}


//count the number of files in a directory with a given ending
int countNumberOfFilesInDirectory(std::string inputDirectory, const char* fileExtension) {
	char search_path[300];
	WIN32_FIND_DATA fd;
	sprintf_s(search_path, fileExtension, inputDirectory.c_str());
	HANDLE hFind = ::FindFirstFile(search_path, &fd);

	//count the number of OCT frames in the folder
	int count = 0;
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{

				count++;

			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return count;
}

//get the path to all models in a directory
void getModelsInDirectory(bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext) {
	bf::directory_iterator end_itr;
	for (bf::directory_iterator itr(dir); itr != end_itr; ++itr) {
		//check that it is a ply file and then add, otherwise ignore..
		std::vector < std::string > strs;
#if BOOST_FILESYSTEM_VERSION == 3
		std::string file = (itr->path().filename()).string();
#else
		std::string file = (itr->path()).filename();
#endif

		boost::split(strs, file, boost::is_any_of("."));
		std::string extension = strs[strs.size() - 1];

		if (extension.compare(ext) == 0)
		{
#if BOOST_FILESYSTEM_VERSION == 3
			std::string path = rel_path_so_far + (itr->path().filename()).string();
#else
			std::string path = rel_path_so_far + (itr->path()).filename();
#endif

			relative_paths.push_back(path);
		}
	}
}

