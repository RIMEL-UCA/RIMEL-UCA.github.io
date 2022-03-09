package fr.rimelj;

import java.util.Date;
import java.util.List;

// import com.fasterxml.jackson.databind.ObjectMapper; // version 2.11.1
// import com.fasterxml.jackson.annotation.JsonProperty; // version 2.11.1
/* ObjectMapper om = new ObjectMapper();
Root root = om.readValue(myJsonString, Root.class); */

public class TerraformModuleDetails{
    public String id;
    public String owner;
    public String namespace;
    public String name;
    public String version;
    public String provider;
    public String provider_logo_url;
    public String description;
    public String source;
    public String tag;
    public Date published_at;
    public int downloads;
    public boolean verified;
    public Root root;
    public List<Object> submodules;
    public List<Object> examples;
    public List<String> providers;
    public List<String> versions;
}


