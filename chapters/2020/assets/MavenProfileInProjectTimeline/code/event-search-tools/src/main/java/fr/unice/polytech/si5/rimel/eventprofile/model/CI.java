package fr.unice.polytech.si5.rimel.eventprofile.model;

public enum CI {
    JENKINS("jenkins"),
    TRAVIS("travis"),
    CIRCLE("circle"),
    ;
    public final String name;

    CI(String s) {
        name = s;
    }
}
