package fr.rimelj;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Input{
    public String name;
    public String type;
    public String description;
    @JsonProperty("default") 
    public String mydefault;
    public boolean required;
}
