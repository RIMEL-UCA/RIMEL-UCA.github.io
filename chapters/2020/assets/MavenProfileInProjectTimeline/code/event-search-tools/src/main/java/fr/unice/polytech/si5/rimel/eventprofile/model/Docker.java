package fr.unice.polytech.si5.rimel.eventprofile.model;

public enum Docker {
    DOCKERFILE("dockerfile"),
    DOCKERCOMPOSE("docker-compose"),
    ;
    public final String name;

    Docker(String s) {
        name = s;
    }
}
