package fr.rimelj;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import com.bertramlabs.plugins.hcl4j.HCLParser;
import com.bertramlabs.plugins.hcl4j.HCLParserException;

public class GoodPractices {

    public void findGoodPractices(String pathString){

    }

    private boolean useVariablesFile(File rootDir){
       return rootDir.list((dir, name) ->  (name.equals("variables.tf") || name.equals("vars.tf"))).length > 0;
    }
    
    private boolean useOutputsFile(File rootDir){
        return rootDir.list((dir, name) ->  name.equals("outputs.tf")).length > 0;
    }

    private boolean useModules(File terraformFile) throws HCLParserException, IOException{
        Map results = new HCLParser().parse(terraformFile, "UTF-8");

        return results.get("module") != null;
    }
}
