package fr.rimelj;

import java.util.List;

public class Root{
    public String path;
    public String name;
    public String readme;
    public boolean empty;
    public List<Input> inputs;
    public List<Output> outputs;
    public List<Object> dependencies;
    public List<ProviderDependency> provider_dependencies;
    public List<Resource> resources;
}
